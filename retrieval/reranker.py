"""
Reranker - Cohere Rerank для улучшения качества retrieval.

Reranking улучшает top-k results после initial retrieval:
- Учитывает контекст query
- Cross-attention между query и документами
- Более точные scores чем cosine similarity

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import cohere

from legal_assistant.core import (
    get_logger,
    get_settings,
    ExternalServiceError,
    track_time,
    track_error,
)
from legal_assistant.utils.retry import retry_on_external_service_error


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class RerankedResult:
    """
    Reranked result.
    
    Attributes:
        index: Original index in input list
        relevance_score: Reranking score (0-1)
        text: Document text
        metadata: Additional metadata
    """
    index: int
    relevance_score: float
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "relevance_score": self.relevance_score,
            "text": self.text,
            "metadata": self.metadata,
        }


class CohereReranker:
    """
    Production-ready Cohere reranker.
    
    Features:
    - Rerank top-k candidates
    - Score threshold filtering
    - Batch processing
    - Error handling
    - Token tracking
    
    Example:
        >>> reranker = CohereReranker()
        >>> results = await reranker.rerank(
        ...     query="договор купли-продажи",
        ...     documents=["doc1", "doc2", "doc3"],
        ...     top_n=2,
        ... )
    """
    
    # Model configuration
    DEFAULT_MODEL = "rerank-multilingual-v3.0"  # Supports Russian
    MAX_DOCUMENTS_PER_REQUEST = 1000
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Initialize Cohere reranker.
        
        Args:
            api_key: Cohere API key.
            model: Reranking model name.
        """
        api_key = api_key or (
            settings.cohere.api_key.get_secret_value()
            if hasattr(settings, "cohere") and settings.cohere.api_key
            else None
        )
        
        if not api_key:
            raise ValueError("Cohere API key required for reranking")
        
        self.model = model
        self.client = cohere.AsyncClient(api_key=api_key)
        
        logger.info(
            "cohere_reranker_initialized",
            model=model,
        )
    
    async def close(self):
        """Close Cohere client."""
        await self.client.close()
        logger.info("cohere_reranker_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @retry_on_external_service_error
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        score_threshold: float = 0.0,
        metadata: Optional[List[Dict[str, Any]]] = None,
        return_documents: bool = True,
    ) -> List[RerankedResult]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: Search query.
            documents: List of document texts to rerank.
            top_n: Return top N results (default: all).
            score_threshold: Minimum relevance score (0-1).
            metadata: Optional metadata for each document.
            return_documents: Include document text in results.
            
        Returns:
            List of reranked results sorted by relevance.
            
        Raises:
            ExternalServiceError: If API call fails.
        """
        if not documents:
            return []
        
        if len(documents) > self.MAX_DOCUMENTS_PER_REQUEST:
            logger.warning(
                "too_many_documents",
                num_docs=len(documents),
                max_allowed=self.MAX_DOCUMENTS_PER_REQUEST,
            )
            documents = documents[:self.MAX_DOCUMENTS_PER_REQUEST]
        
        metadata = metadata or [{}] * len(documents)
        top_n = top_n or len(documents)
        
        logger.info(
            "reranking_documents",
            query_length=len(query),
            num_documents=len(documents),
            top_n=top_n,
        )
        
        try:
            with track_time("reranking_seconds"):
                response = await self.client.rerank(
                    model=self.model,
                    query=query,
                    documents=documents,
                    top_n=top_n,
                    return_documents=return_documents,
                )
            
            # Format results
            results = []
            for result in response.results:
                # Apply score threshold
                if result.relevance_score < score_threshold:
                    continue
                
                results.append(RerankedResult(
                    index=result.index,
                    relevance_score=result.relevance_score,
                    text=documents[result.index] if return_documents else "",
                    metadata=metadata[result.index],
                ))
            
            logger.info(
                "reranking_completed",
                num_results=len(results),
                top_score=results[0].relevance_score if results else 0.0,
            )
            
            return results
        
        except Exception as e:
            track_error("cohere_rerank")
            raise ExternalServiceError(
                f"Cohere rerank error: {e}",
                service="cohere",
                details={
                    "query": query[:100],
                    "num_documents": len(documents),
                    "error": str(e),
                },
            ) from e
    
    async def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        top_n: Optional[int] = None,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents with metadata (convenience method).
        
        Args:
            query: Search query.
            documents: List of dicts with text and metadata.
            text_key: Key for text in document dict.
            top_n: Return top N results.
            score_threshold: Minimum relevance score.
            
        Returns:
            Reranked documents with added 'rerank_score' field.
        """
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Rerank
        results = await self.rerank(
            query=query,
            documents=texts,
            top_n=top_n,
            score_threshold=score_threshold,
            metadata=documents,
            return_documents=False,
        )
        
        # Add rerank scores to original documents
        reranked_docs = []
        for result in results:
            doc = documents[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            doc["original_rank"] = result.index
            reranked_docs.append(doc)
        
        return reranked_docs


# Convenience function
async def rerank_documents(
    query: str,
    documents: List[str],
    top_n: Optional[int] = None,
    **kwargs,
) -> List[RerankedResult]:
    """
    Quick document reranking.
    
    Args:
        query: Search query.
        documents: Document texts.
        top_n: Top N results.
        **kwargs: Passed to CohereReranker.rerank().
        
    Returns:
        Reranked results.
        
    Example:
        >>> results = await rerank_documents(
        ...     query="договор",
        ...     documents=["doc1", "doc2", "doc3"],
        ...     top_n=2,
        ... )
        >>> print(results[0].relevance_score)
    """
    async with CohereReranker() as reranker:
        return await reranker.rerank(query, documents, top_n, **kwargs)
