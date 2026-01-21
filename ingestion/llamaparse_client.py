"""
LlamaParse API Client - Production-Ready Document Parser.

Этот модуль предоставляет wrapper для LlamaParse API с полной поддержкой:
- Retry logic с exponential backoff
- Result caching (Redis)
- Structured logging
- Prometheus metrics
- Error handling с fallback
- Input validation
- Async/await для performance

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import asyncio
import hashlib
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import timedelta

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from legal_assistant.core import (
    get_logger,
    get_settings,
    LegalAssistantException,
    DocumentParsingError,
    ExternalServiceError,
    track_time,
    track_error,
)
from legal_assistant.utils.retry import retry_on_external_service_error


logger = get_logger(__name__)
settings = get_settings()


class LlamaParseError(DocumentParsingError):
    """LlamaParse-specific parsing error."""
    
    def __init__(
        self,
        message: str,
        job_id: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details=details or {})
        self.job_id = job_id
        self.status_code = status_code


class LlamaParseClient:
    """
    Production-ready LlamaParse API client.
    
    Features:
    - Async document parsing
    - Automatic retry logic
    - Result caching (Redis)
    - Comprehensive error handling
    - Prometheus metrics
    - Structured logging
    
    Example:
        >>> client = LlamaParseClient()
        >>> result = await client.parse_document("contract.pdf")
        >>> print(result["text"])
    """
    
    # API Configuration
    BASE_URL = "https://api.cloud.llamaindex.ai/api/parsing"
    UPLOAD_TIMEOUT = 60  # seconds
    STATUS_CHECK_INTERVAL = 2  # seconds
    MAX_STATUS_CHECKS = 30  # 60 seconds total
    
    # Supported file types
    SUPPORTED_FORMATS = {
        "application/pdf": [".pdf"],
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
        "application/msword": [".doc"],
        "text/plain": [".txt"],
    }
    
    def __init__(self, api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize LlamaParse client.
        
        Args:
            api_key: LlamaParse API key. If None, uses config.
            use_cache: Enable Redis caching for parsed results.
        """
        self.api_key = api_key or settings.llamaparse.api_key.get_secret_value()
        self.use_cache = use_cache and settings.llamaparse.enable_caching
        
        # Session будет создана в async context
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Redis cache (если включен)
        self._cache = None
        if self.use_cache:
            try:
                import redis.asyncio as aioredis
                self._cache = aioredis.from_url(
                    settings.redis.url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                logger.warning("Redis not available, caching disabled")
                self.use_cache = False
        
        logger.info(
            "llamaparse_client_initialized",
            use_cache=self.use_cache,
            timeout=self.UPLOAD_TIMEOUT,
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Create aiohttp session if not exists."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.UPLOAD_TIMEOUT)
            self._session = aiohttp.ClientSession(
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "LegalAssistant/1.0",
                },
                timeout=timeout,
            )
    
    async def close(self):
        """Close HTTP session and Redis connection."""
        if self._session and not self._session.closed:
            await self._session.close()
        
        if self._cache:
            await self._cache.close()
        
        logger.info("llamaparse_client_closed")
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate input file.
        
        Args:
            file_path: Path to file.
            
        Raises:
            DocumentParsingError: If file is invalid.
        """
        # Check existence
        if not file_path.exists():
            raise DocumentParsingError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)},
            )
        
        # Check size
        file_size = file_path.stat().st_size
        max_size = settings.llamaparse.max_file_size
        if file_size > max_size:
            raise DocumentParsingError(
                f"File too large: {file_size / 1024 / 1024:.2f}MB (max: {max_size / 1024 / 1024:.2f}MB)",
                details={
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "max_size": max_size,
                },
            )
        
        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type not in self.SUPPORTED_FORMATS:
            supported = ", ".join(
                ext for exts in self.SUPPORTED_FORMATS.values() for ext in exts
            )
            raise DocumentParsingError(
                f"Unsupported file type: {mime_type}. Supported: {supported}",
                details={
                    "file_path": str(file_path),
                    "mime_type": mime_type,
                    "supported_formats": list(self.SUPPORTED_FORMATS.keys()),
                },
            )
        
        logger.debug(
            "file_validated",
            file_path=str(file_path),
            file_size=file_size,
            mime_type=mime_type,
        )
    
    def _get_cache_key(self, file_path: Path, options: Dict[str, Any]) -> str:
        """
        Generate cache key from file content + options.
        
        Args:
            file_path: Path to file.
            options: Parsing options.
            
        Returns:
            Cache key (hex string).
        """
        # Hash file content
        file_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        # Hash options
        options_str = str(sorted(options.items()))
        options_hash = hashlib.sha256(options_str.encode()).hexdigest()
        
        cache_key = f"llamaparse:{file_hash.hexdigest()}:{options_hash}"
        return cache_key
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached parsing result.
        
        Args:
            cache_key: Cache key.
            
        Returns:
            Cached result or None.
        """
        if not self.use_cache or not self._cache:
            return None
        
        try:
            import json
            cached = await self._cache.get(cache_key)
            if cached:
                logger.info("cache_hit", cache_key=cache_key)
                return json.loads(cached)
        except Exception as e:
            logger.warning("cache_get_error", error=str(e))
        
        return None
    
    async def _set_cached_result(
        self,
        cache_key: str,
        result: Dict[str, Any],
        ttl: int = 86400,  # 24 hours
    ) -> None:
        """
        Cache parsing result.
        
        Args:
            cache_key: Cache key.
            result: Parsing result.
            ttl: Time to live (seconds).
        """
        if not self.use_cache or not self._cache:
            return
        
        try:
            import json
            await self._cache.setex(
                cache_key,
                ttl,
                json.dumps(result),
            )
            logger.info("cache_set", cache_key=cache_key, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_error", error=str(e))
    
    @retry_on_external_service_error
    async def _upload_document(
        self,
        file_path: Path,
        options: Dict[str, Any],
    ) -> str:
        """
        Upload document to LlamaParse.
        
        Args:
            file_path: Path to file.
            options: Parsing options.
            
        Returns:
            Job ID.
            
        Raises:
            LlamaParseError: If upload fails.
        """
        await self._ensure_session()
        
        # Prepare form data
        data = aiohttp.FormData()
        data.add_field(
            "file",
            open(file_path, "rb"),
            filename=file_path.name,
            content_type=mimetypes.guess_type(str(file_path))[0],
        )
        
        # Add options
        for key, value in options.items():
            data.add_field(key, str(value))
        
        # Upload
        logger.info(
            "uploading_document",
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
        )
        
        try:
            with track_time("document_upload_seconds"):
                async with self._session.post(
                    f"{self.BASE_URL}/upload",
                    data=data,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise LlamaParseError(
                            f"Upload failed: {error_text}",
                            status_code=response.status,
                            details={
                                "file_path": str(file_path),
                                "status_code": response.status,
                                "response": error_text,
                            },
                        )
                    
                    result = await response.json()
                    job_id = result.get("id")
                    
                    if not job_id:
                        raise LlamaParseError(
                            "No job ID in response",
                            details={"response": result},
                        )
                    
                    logger.info(
                        "document_uploaded",
                        file_path=str(file_path),
                        job_id=job_id,
                    )
                    
                    return job_id
        
        except aiohttp.ClientError as e:
            track_error("llamaparse_upload")
            raise ExternalServiceError(
                f"LlamaParse upload error: {e}",
                service="llamaparse",
                details={"file_path": str(file_path), "error": str(e)},
            ) from e
    
    @retry_on_external_service_error
    async def _check_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check parsing job status.
        
        Args:
            job_id: Job ID.
            
        Returns:
            Status response.
            
        Raises:
            LlamaParseError: If status check fails.
        """
        await self._ensure_session()
        
        try:
            async with self._session.get(
                f"{self.BASE_URL}/job/{job_id}",
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LlamaParseError(
                        f"Status check failed: {error_text}",
                        job_id=job_id,
                        status_code=response.status,
                    )
                
                return await response.json()
        
        except aiohttp.ClientError as e:
            track_error("llamaparse_status_check")
            raise ExternalServiceError(
                f"LlamaParse status check error: {e}",
                service="llamaparse",
                details={"job_id": job_id, "error": str(e)},
            ) from e
    
    async def _wait_for_completion(self, job_id: str) -> Dict[str, Any]:
        """
        Wait for parsing job to complete.
        
        Args:
            job_id: Job ID.
            
        Returns:
            Final status with result.
            
        Raises:
            LlamaParseError: If job fails or times out.
        """
        logger.info("waiting_for_completion", job_id=job_id)
        
        for attempt in range(self.MAX_STATUS_CHECKS):
            status = await self._check_status(job_id)
            state = status.get("status")
            
            logger.debug(
                "job_status_check",
                job_id=job_id,
                attempt=attempt + 1,
                state=state,
            )
            
            if state == "SUCCESS":
                logger.info("job_completed", job_id=job_id)
                return status
            
            elif state == "FAILURE":
                error_msg = status.get("error", "Unknown error")
                track_error("llamaparse_job_failed")
                raise LlamaParseError(
                    f"Parsing job failed: {error_msg}",
                    job_id=job_id,
                    details={"status": status},
                )
            
            elif state in ["PENDING", "PROCESSING"]:
                await asyncio.sleep(self.STATUS_CHECK_INTERVAL)
            
            else:
                logger.warning(
                    "unknown_job_state",
                    job_id=job_id,
                    state=state,
                )
                await asyncio.sleep(self.STATUS_CHECK_INTERVAL)
        
        # Timeout
        track_error("llamaparse_timeout")
        raise LlamaParseError(
            f"Parsing job timeout after {self.MAX_STATUS_CHECKS * self.STATUS_CHECK_INTERVAL}s",
            job_id=job_id,
            details={
                "max_attempts": self.MAX_STATUS_CHECKS,
                "interval": self.STATUS_CHECK_INTERVAL,
            },
        )
    
    async def parse_document(
        self,
        file_path: str | Path,
        *,
        language: str = "ru",
        extract_tables: bool = True,
        extract_images: bool = True,
        preserve_formatting: bool = True,
        custom_instructions: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Parse document using LlamaParse API.
        
        Args:
            file_path: Path to document file.
            language: Document language (default: "ru").
            extract_tables: Extract tables structure.
            extract_images: Extract embedded images.
            preserve_formatting: Preserve text formatting (bold, italic).
            custom_instructions: Custom parsing instructions.
            use_cache: Override instance cache setting.
            
        Returns:
            Parsing result:
            {
                "text": "Full extracted text",
                "metadata": {
                    "num_pages": 10,
                    "language": "ru",
                    "has_tables": True,
                    ...
                },
                "tables": [...],  # If extract_tables=True
                "images": [...],  # If extract_images=True
            }
            
        Raises:
            DocumentParsingError: If file is invalid.
            LlamaParseError: If parsing fails.
            ExternalServiceError: If API is unavailable.
        """
        file_path = Path(file_path)
        use_cache = use_cache if use_cache is not None else self.use_cache
        
        # Validate
        self._validate_file(file_path)
        
        # Parsing options
        options = {
            "language": language,
            "extract_tables": extract_tables,
            "extract_images": extract_images,
            "preserve_formatting": preserve_formatting,
        }
        
        if custom_instructions:
            options["custom_instructions"] = custom_instructions
        
        # Check cache
        cache_key = self._get_cache_key(file_path, options)
        if use_cache:
            cached = await self._get_cached_result(cache_key)
            if cached:
                return cached
        
        # Parse
        logger.info(
            "starting_parse",
            file_path=str(file_path),
            options=options,
        )
        
        with track_time("document_parsing_seconds"):
            # Upload
            job_id = await self._upload_document(file_path, options)
            
            # Wait for completion
            status = await self._wait_for_completion(job_id)
            
            # Extract result
            result = {
                "text": status.get("text", ""),
                "metadata": status.get("metadata", {}),
                "job_id": job_id,
            }
            
            if extract_tables:
                result["tables"] = status.get("tables", [])
            
            if extract_images:
                result["images"] = status.get("images", [])
            
            logger.info(
                "parsing_completed",
                file_path=str(file_path),
                job_id=job_id,
                text_length=len(result["text"]),
                num_tables=len(result.get("tables", [])),
                num_images=len(result.get("images", [])),
            )
            
            # Cache result
            if use_cache:
                await self._set_cached_result(cache_key, result)
            
            return result


# Convenience function for quick usage
async def parse_document(
    file_path: str | Path,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick document parsing (creates and closes client automatically).
    
    Args:
        file_path: Path to document.
        **kwargs: Passed to LlamaParseClient.parse_document().
        
    Returns:
        Parsing result.
        
    Example:
        >>> result = await parse_document("contract.pdf")
        >>> print(result["text"])
    """
    async with LlamaParseClient() as client:
        return await client.parse_document(file_path, **kwargs)
