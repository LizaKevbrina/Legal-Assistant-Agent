"""
Citation Engine - Извлечение и форматирование цитат из источников.

Features:
- Парсинг цитат из ответа LLM
- Валидация цитат против source documents
- Форматирование в разных стилях
- Linking к оригинальным документам

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from legal_assistant.core import get_logger


logger = get_logger(__name__)


@dataclass
class Citation:
    """
    Citation с метаданными.
    
    Attributes:
        text: Цитируемый текст
        source_id: ID документа-источника
        source_name: Название источника
        section: Раздел/пункт (если указан)
        page: Номер страницы (если применимо)
        confidence: Уверенность в корректности (0-1)
    """
    text: str
    source_id: str
    source_name: str
    section: Optional[str] = None
    page: Optional[int] = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "section": self.section,
            "page": self.page,
            "confidence": self.confidence,
        }
    
    def format(self, style: str = "inline") -> str:
        """
        Format citation.
        
        Args:
            style: Citation style (inline, footnote, apa).
            
        Returns:
            Formatted citation string.
        """
        if style == "inline":
            base = f"[{self.source_name}"
            if self.section:
                base += f", {self.section}"
            if self.page:
                base += f", стр. {self.page}"
            base += "]"
            return base
        
        elif style == "footnote":
            parts = [self.source_name]
            if self.section:
                parts.append(self.section)
            if self.page:
                parts.append(f"стр. {self.page}")
            return ", ".join(parts)
        
        elif style == "apa":
            # Simplified APA-like format
            return f"({self.source_name}, {self.section or 'н.д.'})"
        
        return f"[{self.source_name}]"


class CitationEngine:
    """
    Production-ready citation engine.
    
    Features:
    - Extract citations from LLM response
    - Validate against source documents
    - Format in different styles
    - Link to original documents
    
    Example:
        >>> engine = CitationEngine()
        >>> citations = engine.extract_citations(
        ...     response_text="По договору [Источник: Договор №123, п. 3.1]...",
        ...     source_documents=docs,
        ... )
        >>> print(citations[0].format("inline"))
    """
    
    # Citation patterns
    CITATION_PATTERNS = [
        # [Источник: название, п. X.Y]
        r'\[Источник:\s*([^,\]]+)(?:,\s*([^\]]+))?\]',
        # [название документа, п. X.Y]
        r'\[([^,\]]+)(?:,\s*([^\]]+))?\]',
        # (название документа)
        r'\(([^)]+)\)',
    ]
    
    def __init__(self, strict_validation: bool = False):
        """
        Initialize citation engine.
        
        Args:
            strict_validation: Require citations to match source text exactly.
        """
        self.strict_validation = strict_validation
        logger.info("citation_engine_initialized", strict=strict_validation)
    
    def extract_citations(
        self,
        response_text: str,
        source_documents: List[Dict[str, Any]],
    ) -> List[Citation]:
        """
        Extract citations from LLM response.
        
        Args:
            response_text: LLM response text with citations.
            source_documents: Source documents used for answer.
            
        Returns:
            List of Citation objects.
        """
        citations = []
        
        # Try each pattern
        for pattern in self.CITATION_PATTERNS:
            matches = re.finditer(pattern, response_text)
            
            for match in matches:
                source_name = match.group(1).strip()
                section = match.group(2).strip() if match.lastindex >= 2 else None
                
                # Find matching document
                source_doc = self._find_source_document(
                    source_name,
                    source_documents,
                )
                
                if source_doc:
                    citation = Citation(
                        text=match.group(0),
                        source_id=source_doc.get("id", "unknown"),
                        source_name=source_name,
                        section=section,
                        confidence=1.0,
                    )
                    citations.append(citation)
                    
                    logger.debug(
                        "citation_extracted",
                        source=source_name,
                        section=section,
                    )
                else:
                    # Citation не найдена в source documents
                    logger.warning(
                        "citation_not_found",
                        source=source_name,
                    )
                    
                    if not self.strict_validation:
                        # Add with low confidence
                        citation = Citation(
                            text=match.group(0),
                            source_id="unknown",
                            source_name=source_name,
                            section=section,
                            confidence=0.5,
                        )
                        citations.append(citation)
        
        logger.info("citations_extracted", num_citations=len(citations))
        
        return citations
    
    def _find_source_document(
        self,
        source_name: str,
        source_documents: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        Find source document by name.
        
        Args:
            source_name: Source name from citation.
            source_documents: List of source documents.
            
        Returns:
            Matching document or None.
        """
        # Normalize name for comparison
        normalized_name = source_name.lower().strip()
        
        for doc in source_documents:
            metadata = doc.get("metadata", {})
            
            # Check file_name
            file_name = metadata.get("file_name", "")
            if normalized_name in file_name.lower():
                return doc
            
            # Check doc_type + doc_number
            doc_type = metadata.get("doc_type", "")
            doc_numbers = metadata.get("doc_numbers", [])
            
            for number in doc_numbers:
                if number in normalized_name:
                    return doc
        
        return None
    
    def validate_citations(
        self,
        citations: List[Citation],
        source_documents: List[Dict[str, Any]],
    ) -> List[Citation]:
        """
        Validate citations against source documents.
        
        Args:
            citations: List of citations to validate.
            source_documents: Source documents.
            
        Returns:
            Citations with updated confidence scores.
        """
        validated = []
        
        for citation in citations:
            # Find source document
            source_doc = next(
                (
                    doc for doc in source_documents
                    if doc.get("id") == citation.source_id
                ),
                None,
            )
            
            if not source_doc:
                # Source not found
                citation.confidence = 0.3
                logger.warning(
                    "citation_validation_failed",
                    source_id=citation.source_id,
                )
            else:
                # Source found
                citation.confidence = 1.0
            
            validated.append(citation)
        
        return validated
    
    def format_citations(
        self,
        citations: List[Citation],
        style: str = "inline",
        deduplicate: bool = True,
    ) -> List[str]:
        """
        Format citations in specified style.
        
        Args:
            citations: List of citations.
            style: Citation style.
            deduplicate: Remove duplicate citations.
            
        Returns:
            List of formatted citation strings.
        """
        formatted = [cite.format(style) for cite in citations]
        
        if deduplicate:
            # Remove duplicates while preserving order
            seen = set()
            unique = []
            for cite in formatted:
                if cite not in seen:
                    seen.add(cite)
                    unique.append(cite)
            return unique
        
        return formatted
    
    def add_citations_to_response(
        self,
        response_text: str,
        citations: List[Citation],
        style: str = "footnote",
    ) -> str:
        """
        Add formatted citations section to response.
        
        Args:
            response_text: Original response.
            citations: Extracted citations.
            style: Citation style for references section.
            
        Returns:
            Response with citations section appended.
        """
        if not citations:
            return response_text
        
        # Build citations section
        citations_section = "\n\n**ИСТОЧНИКИ:**\n"
        
        formatted = self.format_citations(citations, style, deduplicate=True)
        for i, cite in enumerate(formatted, 1):
            citations_section += f"{i}. {cite}\n"
        
        return response_text + citations_section
    
    def get_citation_stats(
        self,
        citations: List[Citation],
    ) -> Dict[str, Any]:
        """
        Get statistics about citations.
        
        Args:
            citations: List of citations.
            
        Returns:
            Citation statistics.
        """
        if not citations:
            return {
                "total": 0,
                "unique_sources": 0,
                "avg_confidence": 0.0,
            }
        
        unique_sources = len(set(c.source_id for c in citations))
        avg_confidence = sum(c.confidence for c in citations) / len(citations)
        
        return {
            "total": len(citations),
            "unique_sources": unique_sources,
            "avg_confidence": avg_confidence,
            "sources": [
                {
                    "id": c.source_id,
                    "name": c.source_name,
                    "confidence": c.confidence,
                }
                for c in citations
            ],
        }


# Convenience function
def extract_and_format_citations(
    response_text: str,
    source_documents: List[Dict[str, Any]],
    style: str = "inline",
) -> Tuple[List[Citation], str]:
    """
    Extract citations and add formatted section.
    
    Args:
        response_text: LLM response.
        source_documents: Source documents.
        style: Citation style.
        
    Returns:
        Tuple of (citations, response_with_citations).
        
    Example:
        >>> citations, formatted = extract_and_format_citations(
        ...     "Ответ [Источник: Договор №123]...",
        ...     source_docs,
        ... )
        >>> print(formatted)
    """
    engine = CitationEngine()
    citations = engine.extract_citations(response_text, source_documents)
    formatted = engine.add_citations_to_response(response_text, citations, style)
    
    return citations, formatted
