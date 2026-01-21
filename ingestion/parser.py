"""
Document Parser - Главный оркестратор парсинга документов.

Реализует fallback chain:
1. LlamaParse (primary) - лучшее качество
2. GPT-4V (fallback) - для изображений и простых PDF
3. Tesseract OCR (last resort) - надежный, но базовый

Плюс:
- Metadata extraction
- Chunking
- PII detection/redaction
- Error recovery

Автор: AI Legal Assistant Team  
Дата: 2025-01-16
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime

from legal_assistant.core import (
    get_logger,
    get_settings,
    DocumentParsingError,
    track_time,
    track_error,
)
from legal_assistant.ingestion.llamaparse_client import (
    LlamaParseClient,
    LlamaParseError,
)
from legal_assistant.ingestion.vision_parser import (
    VisionParser,
    VisionParserError,
)
from legal_assistant.ingestion.ocr_fallback import (
    OCRFallback,
    OCRError,
)
from legal_assistant.ingestion.metadata_extractor import (
    MetadataExtractor,
    extract_metadata,
)
from legal_assistant.ingestion.chunking import (
    DocumentChunker,
    ChunkingStrategy,
    Chunk,
)


logger = get_logger(__name__)
settings = get_settings()


class ParsingResult:
    """
    Результат парсинга документа.
    
    Attributes:
        text: Извлеченный текст
        metadata: Метаданные документа
        chunks: Разбитый на chunks текст
        parsing_method: Метод парсинга (llamaparse, vision, ocr)
        success: Успешность парсинга
        errors: Список ошибок (если были)
    """
    
    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunks: List[Chunk],
        parsing_method: str,
        success: bool = True,
        errors: Optional[List[str]] = None,
    ):
        self.text = text
        self.metadata = metadata
        self.chunks = chunks
        self.parsing_method = parsing_method
        self.success = success
        self.errors = errors or []
        self.parsed_at = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunks": [c.to_dict() for c in self.chunks],
            "parsing_method": self.parsing_method,
            "success": self.success,
            "errors": self.errors,
            "parsed_at": self.parsed_at,
        }
    
    def __repr__(self) -> str:
        return (
            f"ParsingResult(method={self.parsing_method}, "
            f"text_length={len(self.text)}, "
            f"num_chunks={len(self.chunks)}, "
            f"success={self.success})"
        )


class DocumentParser:
    """
    Production-ready document parser с fallback chain.
    
    Features:
    - Multi-method parsing (LlamaParse → Vision → OCR)
    - Automatic fallback на errors
    - Metadata extraction
    - Semantic chunking
    - PII redaction (опционально)
    - Error recovery
    - Comprehensive logging
    
    Example:
        >>> parser = DocumentParser()
        >>> result = await parser.parse_document("contract.pdf")
        >>> print(result.parsing_method)
        >>> print(result.metadata["doc_type"])
    """
    
    def __init__(
        self,
        use_llamaparse: bool = True,
        use_vision: bool = True,
        use_ocr: bool = True,
        enable_pii_redaction: bool = False,
        chunking_strategy: ChunkingStrategy = ChunkingStrategy.LEGAL_CLAUSE,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize document parser.
        
        Args:
            use_llamaparse: Enable LlamaParse.
            use_vision: Enable GPT-4V fallback.
            use_ocr: Enable Tesseract fallback.
            enable_pii_redaction: Redact PII from text.
            chunking_strategy: Chunking strategy.
            chunk_size: Target chunk size (tokens).
            chunk_overlap: Chunk overlap (tokens).
        """
        self.use_llamaparse = use_llamaparse
        self.use_vision = use_vision
        self.use_ocr = use_ocr
        self.enable_pii_redaction = enable_pii_redaction
        
        # Initialize parsers
        self.llamaparse_client = None
        self.vision_parser = None
        self.ocr_fallback = None
        
        if use_llamaparse:
            self.llamaparse_client = LlamaParseClient()
        
        if use_vision:
            self.vision_parser = VisionParser()
        
        if use_ocr:
            self.ocr_fallback = OCRFallback()
        
        # Metadata extractor
        self.metadata_extractor = MetadataExtractor()
        
        # Chunker
        self.chunker = DocumentChunker(
            strategy=chunking_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        # PII redactor (lazy load)
        self._pii_redactor = None
        
        logger.info(
            "document_parser_initialized",
            use_llamaparse=use_llamaparse,
            use_vision=use_vision,
            use_ocr=use_ocr,
            chunking_strategy=chunking_strategy.value,
        )
    
    async def close(self):
        """Close all parsers."""
        if self.llamaparse_client:
            await self.llamaparse_client.close()
        
        if self.vision_parser:
            await self.vision_parser.close()
        
        logger.info("document_parser_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate input file.
        
        Args:
            file_path: Path to file.
            
        Raises:
            DocumentParsingError: If invalid.
        """
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
                f"File too large: {file_size / 1024 / 1024:.2f}MB",
                details={
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "max_size": max_size,
                },
            )
        
        logger.debug("file_validated", file_path=str(file_path))
    
    def _redact_pii(self, text: str) -> str:
        """
        Redact PII from text using Presidio.
        
        Args:
            text: Original text.
            
        Returns:
            Text with PII redacted.
        """
        if not self.enable_pii_redaction:
            return text
        
        try:
            # Lazy load Presidio
            if self._pii_redactor is None:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
                
                self._pii_redactor = {
                    "analyzer": AnalyzerEngine(),
                    "anonymizer": AnonymizerEngine(),
                }
            
            # Analyze
            results = self._pii_redactor["analyzer"].analyze(
                text=text,
                language="ru",
                entities=[
                    "PERSON",
                    "PHONE_NUMBER",
                    "EMAIL_ADDRESS",
                    "CREDIT_CARD",
                    "IBAN_CODE",
                ],
            )
            
            # Anonymize
            redacted = self._pii_redactor["anonymizer"].anonymize(
                text=text,
                analyzer_results=results,
            )
            
            logger.info(
                "pii_redacted",
                num_entities=len(results),
            )
            
            return redacted.text
        
        except Exception as e:
            logger.warning("pii_redaction_failed", error=str(e))
            return text
    
    async def _try_llamaparse(self, file_path: Path) -> Optional[str]:
        """
        Try parsing with LlamaParse.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Extracted text or None if failed.
        """
        if not self.use_llamaparse or not self.llamaparse_client:
            return None
        
        try:
            logger.info("trying_llamaparse", file_path=str(file_path))
            
            result = await self.llamaparse_client.parse_document(file_path)
            text = result.get("text", "")
            
            if text:
                logger.info(
                    "llamaparse_success",
                    text_length=len(text),
                )
                return text
        
        except LlamaParseError as e:
            logger.warning(
                "llamaparse_failed",
                error=str(e),
                file_path=str(file_path),
            )
            track_error("llamaparse_fallback")
        
        return None
    
    async def _try_vision(self, file_path: Path) -> Optional[str]:
        """
        Try parsing with GPT-4V.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Extracted text or None if failed.
        """
        if not self.use_vision or not self.vision_parser:
            return None
        
        # Only for images and simple PDFs
        if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".pdf"}:
            return None
        
        try:
            logger.info("trying_vision", file_path=str(file_path))
            
            result = await self.vision_parser.parse_image(
                file_path,
                task="ocr",
            )
            text = result.get("text", "")
            
            if text:
                logger.info(
                    "vision_success",
                    text_length=len(text),
                )
                return text
        
        except VisionParserError as e:
            logger.warning(
                "vision_failed",
                error=str(e),
                file_path=str(file_path),
            )
            track_error("vision_fallback")
        
        return None
    
    def _try_ocr(self, file_path: Path) -> Optional[str]:
        """
        Try parsing with OCR.
        
        Args:
            file_path: Path to file.
            
        Returns:
            Extracted text or None if failed.
        """
        if not self.use_ocr or not self.ocr_fallback:
            return None
        
        try:
            logger.info("trying_ocr", file_path=str(file_path))
            
            result = self.ocr_fallback.process_document(file_path)
            text = result.get("text", "")
            
            if text:
                logger.info(
                    "ocr_success",
                    text_length=len(text),
                )
                return text
        
        except OCRError as e:
            logger.warning(
                "ocr_failed",
                error=str(e),
                file_path=str(file_path),
            )
            track_error("ocr_fallback")
        
        return None
    
    async def parse_document(
        self,
        file_path: str | Path,
        document_metadata: Optional[Dict[str, Any]] = None,
    ) -> ParsingResult:
        """
        Parse document with automatic fallback chain.
        
        Args:
            file_path: Path to document.
            document_metadata: Additional metadata (e.g., upload_by, tags).
            
        Returns:
            ParsingResult with text, metadata, chunks.
            
        Raises:
            DocumentParsingError: If all parsing methods fail.
        """
        file_path = Path(file_path)
        document_metadata = document_metadata or {}
        
        # Validate
        self._validate_file(file_path)
        
        logger.info(
            "starting_document_parse",
            file_path=str(file_path),
            file_size=file_path.stat().st_size,
        )
        
        errors = []
        text = None
        parsing_method = None
        
        with track_time("document_parsing_total_seconds"):
            # Try LlamaParse
            text = await self._try_llamaparse(file_path)
            if text:
                parsing_method = "llamaparse"
            else:
                errors.append("LlamaParse failed or skipped")
            
            # Fallback to Vision
            if not text:
                text = await self._try_vision(file_path)
                if text:
                    parsing_method = "vision"
                else:
                    errors.append("Vision failed or skipped")
            
            # Last resort: OCR
            if not text:
                text = self._try_ocr(file_path)
                if text:
                    parsing_method = "ocr"
                else:
                    errors.append("OCR failed or skipped")
            
            # All failed
            if not text:
                raise DocumentParsingError(
                    "All parsing methods failed",
                    details={
                        "file_path": str(file_path),
                        "errors": errors,
                    },
                )
            
            # PII redaction
            if self.enable_pii_redaction:
                text = self._redact_pii(text)
            
            # Extract metadata
            extracted_metadata = self.metadata_extractor.extract(text)
            
            # Combine metadata
            full_metadata = {
                **document_metadata,
                **extracted_metadata,
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix,
                "parsing_method": parsing_method,
            }
            
            # Chunk document
            chunks = self.chunker.chunk_document(text, full_metadata)
            
            logger.info(
                "document_parsed",
                file_path=str(file_path),
                parsing_method=parsing_method,
                text_length=len(text),
                num_chunks=len(chunks),
                doc_type=extracted_metadata.get("doc_type"),
            )
            
            return ParsingResult(
                text=text,
                metadata=full_metadata,
                chunks=chunks,
                parsing_method=parsing_method,
                success=True,
                errors=errors if errors else None,
            )


# Convenience function
async def parse_document(
    file_path: str | Path,
    **kwargs,
) -> ParsingResult:
    """
    Quick document parsing.
    
    Args:
        file_path: Path to document.
        **kwargs: Passed to DocumentParser.
        
    Returns:
        ParsingResult.
        
    Example:
        >>> result = await parse_document("contract.pdf")
        >>> print(f"Method: {result.parsing_method}")
        >>> print(f"Chunks: {len(result.chunks)}")
    """
    async with DocumentParser(**kwargs) as parser:
        return await parser.parse_document(file_path)
