"""
Response Synthesizer - Синтез финального ответа с цитатами.

Полный пайплайн генерации ответа:
1. Получение retrieved documents
2. Построение промпта с контекстом
3. Вызов LLM
4. Извлечение цитат
5. Форматирование ответа

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
)
from legal_assistant.generation.llm_router import LLMRouter, LLMResponse
from legal_assistant.generation.prompts.system import get_system_prompt
from legal_assistant.generation.prompts.qa import build_qa_prompt
from legal_assistant.generation.citation_engine import (
    CitationEngine,
    Citation,
)


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SynthesizedResponse:
    """
    Synthesized response с метаданными.
    
    Attributes:
        answer: Финальный ответ пользователю
        citations: Извлеченные цитаты
        source_documents: Использованные документы
        llm_response: Raw LLM response
        confidence: Уверенность в ответе (0-1)
    """
    answer: str
    citations: List[Citation]
    source_documents: List[Dict[str, Any]]
    llm_response: LLMResponse
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "num_sources": len(self.source_documents),
            "confidence": self.confidence,
            "llm_metadata": {
                "model": self.llm_response.model,
                "provider": self.llm_response.provider,
                "tokens": self.llm_response.total_tokens,
                "cost_usd": self.llm_response.cost_usd,
            },
        }


class ResponseSynthesizer:
    """
    Production-ready response synthesizer.
    
    Combines:
    - LLM routing
    - Prompt engineering
    - Citation extraction
    - Response formatting
    
    Example:
        >>> synthesizer = ResponseSynthesizer()
        >>> response = await synthesizer.synthesize(
        ...     query="Что такое договор?",
        ...     retrieved_documents=docs,
        ... )
        >>> print(response.answer)
    """
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        task_type: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        citation_style: str = "inline",
    ):
        """
        Initialize response synthesizer.
        
        Args:
            model: LLM model to use.
            task_type: Task type for system prompt.
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.
            citation_style: Citation formatting style.
        """
        self.model = model
        self.task_type = task_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.citation_style = citation_style
        
        # Components
        self.llm_router = LLMRouter(primary_model=model)
        self.citation_engine = CitationEngine()
        
        # System prompt
        self.system_prompt = get_system_prompt(task_type)
        
        logger.info(
            "response_synthesizer_initialized",
            model=model,
            task_type=task_type,
        )
    
    async def close(self):
        """Close LLM router."""
        await self.llm_router.close()
        logger.info("response_synthesizer_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _build_messages(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """
        Build messages for LLM.
        
        Args:
            query: User query.
            retrieved_documents: Retrieved documents.
            
        Returns:
            Messages list (OpenAI format).
        """
        # Build user prompt
        user_prompt = build_qa_prompt(
            query=query,
            context_documents=retrieved_documents,
            include_sources=True,
        )
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        return messages
    
    def _calculate_confidence(
        self,
        citations: List[Citation],
        retrieved_documents: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate confidence score for answer.
        
        Args:
            citations: Extracted citations.
            retrieved_documents: Source documents.
            
        Returns:
            Confidence score (0-1).
        """
        if not citations:
            # No citations = low confidence
            return 0.5
        
        # Average citation confidence
        avg_citation_confidence = sum(c.confidence for c in citations) / len(citations)
        
        # Number of unique sources used
        unique_sources = len(set(c.source_id for c in citations))
        source_coverage = min(unique_sources / len(retrieved_documents), 1.0) if retrieved_documents else 0
        
        # Combined confidence
        confidence = (avg_citation_confidence * 0.7) + (source_coverage * 0.3)
        
        return confidence
    
    async def synthesize(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> SynthesizedResponse:
        """
        Synthesize final response.
        
        Args:
            query: User query.
            retrieved_documents: Retrieved documents from RAG.
            conversation_history: Previous conversation (for follow-up).
            
        Returns:
            SynthesizedResponse with answer and metadata.
        """
        logger.info(
            "synthesizing_response",
            query_length=len(query),
            num_docs=len(retrieved_documents),
        )
        
        with track_time("response_synthesis_seconds"):
            # Build messages
            messages = self._build_messages(query, retrieved_documents)
            
            # Add conversation history if provided
            if conversation_history:
                # Insert history before final user message
                messages = [messages[0]] + conversation_history + [messages[-1]]
            
            # Call LLM
            llm_response = await self.llm_router.complete(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Extract citations
            citations = self.citation_engine.extract_citations(
                response_text=llm_response.content,
                source_documents=retrieved_documents,
            )
            
            # Validate citations
            citations = self.citation_engine.validate_citations(
                citations=citations,
                source_documents=retrieved_documents,
            )
            
            # Add citations section
            answer_with_citations = self.citation_engine.add_citations_to_response(
                response_text=llm_response.content,
                citations=citations,
                style=self.citation_style,
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                citations=citations,
                retrieved_documents=retrieved_documents,
            )
        
        logger.info(
            "response_synthesized",
            num_citations=len(citations),
            confidence=confidence,
            tokens=llm_response.total_tokens,
        )
        
        return SynthesizedResponse(
            answer=answer_with_citations,
            citations=citations,
            source_documents=retrieved_documents,
            llm_response=llm_response,
            confidence=confidence,
        )
    
    async def synthesize_with_task(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        task_type: str,
    ) -> SynthesizedResponse:
        """
        Synthesize with specific task type.
        
        Args:
            query: User query.
            retrieved_documents: Retrieved documents.
            task_type: Task type (contract_analysis, risk_assessment, etc.).
            
        Returns:
            SynthesizedResponse.
        """
        # Temporarily override system prompt
        original_system_prompt = self.system_prompt
        self.system_prompt = get_system_prompt(task_type)
        
        try:
            response = await self.synthesize(query, retrieved_documents)
            return response
        finally:
            # Restore original
            self.system_prompt = original_system_prompt
    
    def get_synthesis_stats(self) -> Dict[str, Any]:
        """
        Get synthesis statistics.
        
        Returns:
            Usage statistics.
        """
        return self.llm_router.get_usage_stats()


# Convenience function
async def synthesize_response(
    query: str,
    retrieved_documents: List[Dict[str, Any]],
    model: str = "gpt-4-turbo",
    **kwargs,
) -> SynthesizedResponse:
    """
    Quick response synthesis.
    
    Args:
        query: User query.
        retrieved_documents: Retrieved documents.
        model: LLM model.
        **kwargs: Additional synthesizer params.
        
    Returns:
        SynthesizedResponse.
        
    Example:
        >>> response = await synthesize_response(
        ...     "Что такое договор купли-продажи?",
        ...     retrieved_docs,
        ... )
        >>> print(response.answer)
        >>> print(f"Confidence: {response.confidence}")
    """
    async with ResponseSynthesizer(model=model, **kwargs) as synthesizer:
        return await synthesizer.synthesize(query, retrieved_documents)
