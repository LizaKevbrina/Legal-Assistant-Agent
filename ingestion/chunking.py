"""
Semantic Chunking - Умная разбивка документов на chunks.

Стратегии:
- SentenceWindowChunker: Chunks с контекстом соседних предложений
- SemanticChunker: Разбивка по семантической схожести
- LegalClauseChunker: Сохранение границ юридических пунктов
- HierarchicalChunker: Многоуровневая разбивка (секции → параграфы → предложения)

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
)


logger = get_logger(__name__)
settings = get_settings()


class ChunkingStrategy(str, Enum):
    """Стратегии разбивки."""
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC = "semantic"
    LEGAL_CLAUSE = "legal_clause"
    HIERARCHICAL = "hierarchical"
    FIXED_SIZE = "fixed_size"


@dataclass
class Chunk:
    """
    Chunk документа с метаданными.
    
    Attributes:
        text: Текст chunk'а
        metadata: Метаданные (page, section, etc.)
        start_char: Позиция начала в исходном тексте
        end_char: Позиция конца в исходном тексте
    """
    text: str
    metadata: Dict[str, Any]
    start_char: int
    end_char: int
    
    def __len__(self) -> int:
        """Length in characters."""
        return len(self.text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class BaseChunker:
    """Base class для chunkers."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target chunk size (в токенах, примерно).
            chunk_overlap: Overlap между chunks (в токенах).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Approximation: 1 token ≈ 4 chars for Russian
        self.char_size = chunk_size * 4
        self.char_overlap = chunk_overlap * 4
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count (approximation).
        
        Args:
            text: Text string.
            
        Returns:
            Approximate token count.
        """
        return len(text) // 4
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text.
        
        Args:
            text: Text to chunk.
            metadata: Optional metadata to attach to all chunks.
            
        Returns:
            List of Chunk objects.
        """
        raise NotImplementedError


class SentenceWindowChunker(BaseChunker):
    """
    Chunker с контекстом соседних предложений.
    
    Каждый chunk содержит:
    - Основные предложения (window_size)
    - Предыдущие предложения для контекста
    - Следующие предложения для контекста
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        window_size: int = 3,
    ):
        """
        Initialize sentence window chunker.
        
        Args:
            chunk_size: Target chunk size in tokens.
            chunk_overlap: Overlap in tokens.
            window_size: Number of sentences per main window.
        """
        super().__init__(chunk_size, chunk_overlap)
        self.window_size = window_size
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text.
            
        Returns:
            List of sentences.
        """
        # Regex для разбивки по предложениям (упрощенный)
        # Учитывает точки, восклицательные и вопросительные знаки
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text with sentence windows.
        
        Args:
            text: Text to chunk.
            metadata: Metadata for chunks.
            
        Returns:
            List of chunks.
        """
        logger.debug("chunking_with_sentence_window", text_length=len(text))
        
        metadata = metadata or {}
        sentences = self._split_sentences(text)
        chunks = []
        
        # Slide window
        i = 0
        while i < len(sentences):
            # Main window
            window_sentences = sentences[i:i + self.window_size]
            
            # Context (overlap with previous)
            prev_context = sentences[max(0, i - 1):i]
            
            # Context (overlap with next)
            next_context = sentences[i + self.window_size:i + self.window_size + 1]
            
            # Combine
            all_sentences = prev_context + window_sentences + next_context
            chunk_text = " ".join(all_sentences)
            
            # Find positions
            start_char = text.find(window_sentences[0])
            end_char = text.find(window_sentences[-1]) + len(window_sentences[-1])
            
            # Create chunk
            chunk = Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "sentence_start": i,
                    "sentence_end": i + len(window_sentences),
                    "has_prev_context": len(prev_context) > 0,
                    "has_next_context": len(next_context) > 0,
                },
                start_char=start_char,
                end_char=end_char,
            )
            
            chunks.append(chunk)
            
            # Move window (with overlap consideration)
            i += max(1, self.window_size - 1)
        
        logger.info(
            "chunking_completed",
            num_chunks=len(chunks),
            num_sentences=len(sentences),
        )
        
        return chunks


class LegalClauseChunker(BaseChunker):
    """
    Chunker для юридических документов.
    
    Сохраняет границы:
    - Статей и пунктов
    - Numbered clauses (1.1, 1.2, etc.)
    - Sections
    """
    
    # Patterns для пунктов
    CLAUSE_PATTERNS = [
        r"^\d+\.\d+\.?",  # 1.1, 2.3
        r"^п\.\s*\d+",  # п. 1
        r"^Статья\s+\d+",  # Статья 1
        r"^Раздел\s+\d+",  # Раздел 1
    ]
    
    def _split_by_clauses(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text by legal clauses.
        
        Args:
            text: Input text.
            
        Returns:
            List of clauses with metadata.
        """
        lines = text.split("\n")
        clauses = []
        current_clause = []
        current_number = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if starts new clause
            is_clause_start = False
            for pattern in self.CLAUSE_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    is_clause_start = True
                    
                    # Save previous clause
                    if current_clause:
                        clauses.append({
                            "text": "\n".join(current_clause),
                            "clause_number": current_number,
                        })
                    
                    # Start new clause
                    current_clause = [line]
                    current_number = match.group(0)
                    break
            
            if not is_clause_start:
                current_clause.append(line)
        
        # Add last clause
        if current_clause:
            clauses.append({
                "text": "\n".join(current_clause),
                "clause_number": current_number,
            })
        
        return clauses
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text preserving legal clause boundaries.
        
        Args:
            text: Text to chunk.
            metadata: Metadata for chunks.
            
        Returns:
            List of chunks.
        """
        logger.debug("chunking_with_legal_clauses", text_length=len(text))
        
        metadata = metadata or {}
        clauses = self._split_by_clauses(text)
        chunks = []
        
        current_chunk_text = []
        current_chunk_tokens = 0
        
        for i, clause in enumerate(clauses):
            clause_text = clause["text"]
            clause_tokens = self._estimate_tokens(clause_text)
            
            # If single clause is too large, split it
            if clause_tokens > self.chunk_size:
                # Save current chunk if any
                if current_chunk_text:
                    chunk_text = "\n".join(current_chunk_text)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "chunking_strategy": "legal_clause",
                        },
                        start_char=text.find(chunk_text),
                        end_char=text.find(chunk_text) + len(chunk_text),
                    ))
                    current_chunk_text = []
                    current_chunk_tokens = 0
                
                # Use sentence chunker for large clause
                sentence_chunker = SentenceWindowChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                sub_chunks = sentence_chunker.chunk(clause_text, metadata)
                chunks.extend(sub_chunks)
            
            # Add clause to current chunk
            elif current_chunk_tokens + clause_tokens <= self.chunk_size:
                current_chunk_text.append(clause_text)
                current_chunk_tokens += clause_tokens
            
            # Start new chunk
            else:
                # Save current chunk
                if current_chunk_text:
                    chunk_text = "\n".join(current_chunk_text)
                    chunks.append(Chunk(
                        text=chunk_text,
                        metadata={
                            **metadata,
                            "chunk_index": len(chunks),
                            "chunking_strategy": "legal_clause",
                        },
                        start_char=text.find(chunk_text),
                        end_char=text.find(chunk_text) + len(chunk_text),
                    ))
                
                # Start new chunk with current clause
                current_chunk_text = [clause_text]
                current_chunk_tokens = clause_tokens
        
        # Add last chunk
        if current_chunk_text:
            chunk_text = "\n".join(current_chunk_text)
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunking_strategy": "legal_clause",
                },
                start_char=text.find(chunk_text),
                end_char=text.find(chunk_text) + len(chunk_text),
            ))
        
        logger.info(
            "chunking_completed",
            num_chunks=len(chunks),
            num_clauses=len(clauses),
        )
        
        return chunks


class FixedSizeChunker(BaseChunker):
    """Simple fixed-size chunker с overlap."""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Chunk text into fixed-size pieces.
        
        Args:
            text: Text to chunk.
            metadata: Metadata for chunks.
            
        Returns:
            List of chunks.
        """
        logger.debug("chunking_with_fixed_size", text_length=len(text))
        
        metadata = metadata or {}
        chunks = []
        
        start = 0
        while start < len(text):
            end = start + self.char_size
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunking_strategy": "fixed_size",
                },
                start_char=start,
                end_char=min(end, len(text)),
            ))
            
            # Move with overlap
            start += self.char_size - self.char_overlap
        
        logger.info("chunking_completed", num_chunks=len(chunks))
        
        return chunks


class DocumentChunker:
    """
    Main document chunker с поддержкой разных стратегий.
    
    Example:
        >>> chunker = DocumentChunker(strategy="legal_clause")
        >>> chunks = chunker.chunk_document(text)
    """
    
    def __init__(
        self,
        strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_WINDOW,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        **kwargs,
    ):
        """
        Initialize document chunker.
        
        Args:
            strategy: Chunking strategy.
            chunk_size: Target chunk size (tokens).
            chunk_overlap: Overlap (tokens).
            **kwargs: Strategy-specific parameters.
        """
        self.strategy = strategy
        
        # Select chunker
        if strategy == ChunkingStrategy.SENTENCE_WINDOW:
            self.chunker = SentenceWindowChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **kwargs,
            )
        elif strategy == ChunkingStrategy.LEGAL_CLAUSE:
            self.chunker = LegalClauseChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        elif strategy == ChunkingStrategy.FIXED_SIZE:
            self.chunker = FixedSizeChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(
            "document_chunker_initialized",
            strategy=strategy.value,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    
    def chunk_document(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Chunk]:
        """
        Chunk document.
        
        Args:
            text: Document text.
            metadata: Document metadata.
            
        Returns:
            List of chunks.
        """
        with track_time("document_chunking_seconds"):
            chunks = self.chunker.chunk(text, metadata)
        
        return chunks


# Convenience function
def chunk_document(
    text: str,
    strategy: ChunkingStrategy = ChunkingStrategy.SENTENCE_WINDOW,
    **kwargs,
) -> List[Chunk]:
    """
    Quick document chunking.
    
    Args:
        text: Document text.
        strategy: Chunking strategy.
        **kwargs: Passed to DocumentChunker.
        
    Returns:
        List of chunks.
        
    Example:
        >>> chunks = chunk_document(text, strategy="legal_clause")
        >>> for chunk in chunks:
        ...     print(chunk.text[:100])
    """
    chunker = DocumentChunker(strategy=strategy, **kwargs)
    return chunker.chunk_document(text)
