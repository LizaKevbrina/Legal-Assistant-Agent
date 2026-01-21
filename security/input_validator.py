"""
Input validation and sanitization.

Protects against:
- SQL injection
- XSS attacks
- Path traversal
- File upload attacks
- Query length abuse
- Invalid file types

Production features:
- Comprehensive validation rules
- Custom validators
- Error reporting
- Performance optimized
"""

import re
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from enum import Enum

from pydantic import BaseModel, validator, Field

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_error,
)
from legal_assistant.core.exceptions import (
    FileValidationError,
    QueryValidationError,
    ValidationError,
)

logger = get_logger(__name__)
settings = get_settings()


class FileType(str, Enum):
    """Supported file types."""
    
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    DOC = "application/msword"
    TXT = "text/plain"
    JPEG = "image/jpeg"
    PNG = "image/png"


class SanitizationMode(str, Enum):
    """Text sanitization modes."""
    
    STRICT = "strict"      # Remove all special chars
    MODERATE = "moderate"  # Allow common punctuation
    LENIENT = "lenient"    # Minimal sanitization


class InputValidator:
    """
    Input validation and sanitization.
    
    Features:
    - File validation (size, type, content)
    - Query validation (length, SQL injection)
    - Path validation (traversal attacks)
    - Text sanitization (XSS)
    
    Example:
        >>> validator = InputValidator()
        >>> validator.validate_file(Path("contract.pdf"), max_size_mb=50)
        >>> validator.validate_query("Что такое договор?")
        >>> sanitized = validator.sanitize_text("<script>alert('xss')</script>")
    """
    
    # SQL injection patterns
    SQL_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"('.*--)",
        r"(UNION.*SELECT)",
    ]
    
    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]
    
    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\%2f",
        r"%2e%2e/",
        r"\.\.\\",
    ]
    
    def __init__(self):
        """Initialize validator."""
        logger.info("input_validator_initialized")
    
    # ==================== FILE VALIDATION ====================
    
    def validate_file(
        self,
        file_path: Path,
        max_size_mb: Optional[float] = None,
        allowed_types: Optional[Set[FileType]] = None,
        check_content: bool = True,
    ) -> Dict[str, Any]:
        """
        Validate file before processing.
        
        Args:
            file_path: Path to file
            max_size_mb: Max file size in MB (None = use settings)
            allowed_types: Allowed MIME types (None = use defaults)
            check_content: Verify file content matches extension
        
        Returns:
            File metadata dict
        
        Raises:
            FileValidationError: If validation fails
        
        Example:
            >>> validator.validate_file(
            ...     Path("contract.pdf"),
            ...     max_size_mb=50,
            ...     allowed_types={FileType.PDF}
            ... )
            {'size_mb': 2.5, 'mime_type': 'application/pdf', 'extension': '.pdf'}
        """
        try:
            # Check file exists
            if not file_path.exists():
                raise FileValidationError(
                    message="File does not exist",
                    details={"path": str(file_path)},
                )
            
            if not file_path.is_file():
                raise FileValidationError(
                    message="Path is not a file",
                    details={"path": str(file_path)},
                )
            
            # Check file size
            max_size = max_size_mb or settings.ingestion.max_file_size_mb
            size_mb = file_path.stat().st_size / (1024 * 1024)
            
            if size_mb > max_size:
                raise FileValidationError(
                    message=f"File size exceeds {max_size}MB",
                    details={
                        "path": str(file_path),
                        "size_mb": round(size_mb, 2),
                        "max_size_mb": max_size,
                    },
                )
            
            # Detect MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            extension = file_path.suffix.lower()
            
            if mime_type is None:
                raise FileValidationError(
                    message="Cannot determine file type",
                    details={"path": str(file_path), "extension": extension},
                )
            
            # Check allowed types
            allowed = allowed_types or {
                FileType.PDF,
                FileType.DOCX,
                FileType.DOC,
                FileType.TXT,
            }
            
            if mime_type not in {t.value for t in allowed}:
                raise FileValidationError(
                    message="File type not allowed",
                    details={
                        "path": str(file_path),
                        "mime_type": mime_type,
                        "allowed_types": [t.value for t in allowed],
                    },
                )
            
            # Content validation (magic bytes)
            if check_content:
                self._validate_file_content(file_path, mime_type)
            
            # Check filename (no path traversal)
            self.validate_filename(file_path.name)
            
            logger.debug(
                "file_validated",
                path=str(file_path),
                size_mb=round(size_mb, 2),
                mime_type=mime_type,
            )
            
            return {
                "size_mb": round(size_mb, 2),
                "mime_type": mime_type,
                "extension": extension,
                "name": file_path.name,
            }
        
        except FileValidationError:
            raise
        except Exception as e:
            track_error("security", e)
            logger.exception("file_validation_failed", path=str(file_path))
            raise FileValidationError(
                message="File validation failed",
                details={"path": str(file_path), "error": str(e)},
            )
    
    def _validate_file_content(self, file_path: Path, expected_mime: str):
        """Validate file content matches MIME type (magic bytes)."""
        # PDF magic bytes
        if expected_mime == FileType.PDF.value:
            with open(file_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF-"):
                    raise FileValidationError(
                        message="File content does not match PDF format",
                        details={"path": str(file_path)},
                    )
        
        # DOCX magic bytes (ZIP format)
        elif expected_mime == FileType.DOCX.value:
            with open(file_path, "rb") as f:
                header = f.read(4)
                if not header.startswith(b"PK\x03\x04"):
                    raise FileValidationError(
                        message="File content does not match DOCX format",
                        details={"path": str(file_path)},
                    )
        
        # DOC magic bytes
        elif expected_mime == FileType.DOC.value:
            with open(file_path, "rb") as f:
                header = f.read(8)
                if not header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
                    raise FileValidationError(
                        message="File content does not match DOC format",
                        details={"path": str(file_path)},
                    )
    
    def validate_filename(self, filename: str) -> str:
        """
        Validate filename (no path traversal).
        
        Args:
            filename: Filename to validate
        
        Returns:
            Validated filename
        
        Raises:
            FileValidationError: If filename contains invalid chars
        """
        # Check for path traversal
        for pattern in self.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, filename, re.IGNORECASE):
                raise FileValidationError(
                    message="Filename contains path traversal attempt",
                    details={"filename": filename},
                )
        
        # Check for null bytes
        if "\x00" in filename:
            raise FileValidationError(
                message="Filename contains null byte",
                details={"filename": filename},
            )
        
        # Check length
        if len(filename) > 255:
            raise FileValidationError(
                message="Filename too long (max 255 chars)",
                details={"filename": filename, "length": len(filename)},
            )
        
        # Check for dangerous chars
        dangerous_chars = ["<", ">", ":", '"', "|", "?", "*"]
        for char in dangerous_chars:
            if char in filename:
                raise FileValidationError(
                    message=f"Filename contains invalid character: {char}",
                    details={"filename": filename},
                )
        
        return filename
    
    # ==================== QUERY VALIDATION ====================
    
    def validate_query(
        self,
        query: str,
        max_length: Optional[int] = None,
        check_sql_injection: bool = True,
    ) -> str:
        """
        Validate user query.
        
        Args:
            query: User query text
            max_length: Max query length (None = use settings)
            check_sql_injection: Check for SQL injection
        
        Returns:
            Validated query
        
        Raises:
            QueryValidationError: If validation fails
        
        Example:
            >>> validator.validate_query("Что такое договор?")
            'Что такое договор?'
            >>> validator.validate_query("' OR 1=1--")
            # Raises QueryValidationError
        """
        try:
            # Check empty
            if not query or not query.strip():
                raise QueryValidationError(
                    message="Query is empty",
                    details={"query": query},
                )
            
            # Check length
            max_len = max_length or settings.ingestion.max_query_length
            if len(query) > max_len:
                raise QueryValidationError(
                    message=f"Query exceeds {max_len} characters",
                    details={
                        "query": query[:100] + "...",
                        "length": len(query),
                        "max_length": max_len,
                    },
                )
            
            # Check for SQL injection
            if check_sql_injection:
                for pattern in self.SQL_PATTERNS:
                    if re.search(pattern, query, re.IGNORECASE):
                        logger.warning(
                            "sql_injection_attempt_detected",
                            query=query[:100],
                        )
                        raise QueryValidationError(
                            message="Query contains SQL injection pattern",
                            details={"pattern": pattern},
                        )
            
            # Check for excessive whitespace
            if len(query.split()) > 500:
                raise QueryValidationError(
                    message="Query contains too many words (max 500)",
                    details={"word_count": len(query.split())},
                )
            
            logger.debug("query_validated", query_length=len(query))
            
            return query.strip()
        
        except QueryValidationError:
            raise
        except Exception as e:
            track_error("security", e)
            logger.exception("query_validation_failed")
            raise QueryValidationError(
                message="Query validation failed",
                details={"error": str(e)},
            )
    
    # ==================== TEXT SANITIZATION ====================
    
    def sanitize_text(
        self,
        text: str,
        mode: SanitizationMode = SanitizationMode.MODERATE,
        remove_xss: bool = True,
    ) -> str:
        """
        Sanitize text (remove XSS, special chars).
        
        Args:
            text: Input text
            mode: Sanitization strictness
            remove_xss: Remove XSS patterns
        
        Returns:
            Sanitized text
        
        Example:
            >>> validator.sanitize_text("<script>alert('xss')</script>Hello")
            'Hello'
            >>> validator.sanitize_text("Text with\x00null byte")
            'Text with null byte'
        """
        sanitized = text
        
        # Remove XSS patterns
        if remove_xss:
            for pattern in self.XSS_PATTERNS:
                sanitized = re.sub(pattern, "", sanitized, flags=re.IGNORECASE)
        
        # Remove null bytes
        sanitized = sanitized.replace("\x00", "")
        
        # Remove control characters (except newline, tab)
        sanitized = "".join(
            char for char in sanitized
            if char.isprintable() or char in ["\n", "\t", "\r"]
        )
        
        # Mode-specific sanitization
        if mode == SanitizationMode.STRICT:
            # Keep only alphanumeric, space, basic punctuation
            sanitized = re.sub(r"[^a-zA-Zа-яА-Я0-9\s.,!?;:()\-]", "", sanitized)
        
        elif mode == SanitizationMode.MODERATE:
            # Remove most special chars, keep common punctuation
            sanitized = re.sub(r"[^\w\s.,!?;:()\-'\"№]", "", sanitized, flags=re.UNICODE)
        
        # LENIENT: minimal sanitization (already done above)
        
        # Normalize whitespace
        sanitized = " ".join(sanitized.split())
        
        logger.debug(
            "text_sanitized",
            original_length=len(text),
            sanitized_length=len(sanitized),
            mode=mode,
        )
        
        return sanitized
    
    # ==================== METADATA VALIDATION ====================
    
    def validate_metadata(
        self,
        metadata: Dict[str, Any],
        required_fields: Optional[List[str]] = None,
        allowed_fields: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Validate metadata dictionary.
        
        Args:
            metadata: Metadata to validate
            required_fields: Required field names
            allowed_fields: Allowed field names (None = all)
        
        Returns:
            Validated metadata
        
        Raises:
            ValidationError: If validation fails
        """
        # Check required fields
        if required_fields:
            missing = set(required_fields) - set(metadata.keys())
            if missing:
                raise ValidationError(
                    message="Missing required metadata fields",
                    details={"missing_fields": list(missing)},
                )
        
        # Check allowed fields
        if allowed_fields:
            extra = set(metadata.keys()) - allowed_fields
            if extra:
                raise ValidationError(
                    message="Metadata contains disallowed fields",
                    details={"extra_fields": list(extra)},
                )
        
        # Sanitize string values
        sanitized = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                sanitized[key] = self.sanitize_text(value, mode=SanitizationMode.MODERATE)
            else:
                sanitized[key] = value
        
        logger.debug("metadata_validated", field_count=len(sanitized))
        
        return sanitized
    
    # ==================== PAGINATION VALIDATION ====================
    
    def validate_pagination(
        self,
        limit: int,
        offset: int,
        max_limit: int = 100,
    ) -> tuple[int, int]:
        """
        Validate pagination parameters.
        
        Args:
            limit: Items per page
            offset: Page offset
            max_limit: Maximum allowed limit
        
        Returns:
            Validated (limit, offset)
        
        Raises:
            ValidationError: If validation fails
        """
        # Validate limit
        if limit < 1:
            raise ValidationError(
                message="Limit must be >= 1",
                details={"limit": limit},
            )
        
        if limit > max_limit:
            raise ValidationError(
                message=f"Limit exceeds maximum ({max_limit})",
                details={"limit": limit, "max_limit": max_limit},
            )
        
        # Validate offset
        if offset < 0:
            raise ValidationError(
                message="Offset must be >= 0",
                details={"offset": offset},
            )
        
        return limit, offset


# Pydantic models for validation

class FileUploadRequest(BaseModel):
    """File upload request validation."""
    
    filename: str = Field(..., max_length=255)
    size_bytes: int = Field(..., ge=1)
    mime_type: str
    
    @validator("filename")
    def validate_filename(cls, v):
        validator = InputValidator()
        return validator.validate_filename(v)
    
    @validator("size_bytes")
    def validate_size(cls, v):
        max_size_bytes = settings.ingestion.max_file_size_mb * 1024 * 1024
        if v > max_size_bytes:
            raise ValueError(f"File size exceeds {settings.ingestion.max_file_size_mb}MB")
        return v


class QueryRequest(BaseModel):
    """Query request validation."""
    
    query: str = Field(..., min_length=1, max_length=1000)
    
    @validator("query")
    def validate_query(cls, v):
        validator = InputValidator()
        return validator.validate_query(v)


class PaginationParams(BaseModel):
    """Pagination parameters validation."""
    
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)


# Convenience functions

def validate_file(file_path: Path, **kwargs) -> Dict[str, Any]:
    """Convenience function for file validation."""
    validator = InputValidator()
    return validator.validate_file(file_path, **kwargs)


def validate_query(query: str, **kwargs) -> str:
    """Convenience function for query validation."""
    validator = InputValidator()
    return validator.validate_query(query, **kwargs)


def sanitize_text(text: str, **kwargs) -> str:
    """Convenience function for text sanitization."""
    validator = InputValidator()
    return validator.sanitize_text(text, **kwargs)
