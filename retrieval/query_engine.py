"""
Query Engine - LlamaIndex QueryEngine для RAG pipeline.

Полный RAG pipeline:
1. Query embedding
2. Vector search (Qdrant)
3. Hybrid search (optional)
4. Reranking (Cohere)
5. Context compression
6. LLM generation с citations

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
    track_error,
)
from legal_assistant.retrieval.vector_store import QdrantVectorStore
from legal_assistant.retrieval.embeddings import EmbeddingGenerator
from legal_assistant.retrieval.hybrid_search import HybridSearch
from legal_assistant.retrieval.reranker import CohereReranker


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class QueryResult:
    """
    Query result с retrieved documents.
    
    Attributes:
        query: Original query
        retrieved_docs: Retrieved documents
        num_docs_retrieved: Number of docs after retrieval
        num_docs_after_rerank: Number after reranking
    """
    query: str
    retrieved_docs: List[Dict[str, Any]]
    num_docs_retrieved: int
    num_docs_after_rerank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "retrieved_docs": self.retrieved_docs,
            "num_docs_retrieved": self.num_docs_retrieved,
            "num_docs_after_rerank": self.num_docs_after_rerank,
        }


class RAGQueryEngine:
    """
    Production-ready RAG Query Engine.
    
    Pipeline:
    1. Embed query
    2. Search vectors (Qdrant)
    3. Rerank results (Cohere)
    4. Return top-k contexts
    
    Example:
        >>> engine = RAGQueryEngine(collection_name="legal_docs")
        >>> result = await engine.query(
        ...     "Что такое договор купли-продажи?",
        ...     top_k=5,
        ... )
        >>> print(result.retrieved_docs[0]["text"])
    """
    
    def __init__(
        self,
        collection_name: str,
        embedding_model: str = "text-embedding-3-large",
        use_reranking: bool = True,
        rerank_top_n: Optional[int] = None,
        score_threshold: float = 0.5,
    ):
        """
        Initialize RAG query engine.
        
        Args:
            collection_name: Qdrant collection name.
            embedding_model: Embedding model for queries.
            use_reranking: Enable Cohere reranking.
            rerank_top_n: Rerank top N (default: same as retrieval).
            score_threshold: Minimum similarity score.
        """
        self.collection_name = collection_name
        self.use_reranking = use_reranking
        self.rerank_top_n = rerank_top_n
        self.score_threshold = score_threshold
        
        # Components
        self.vector_store = QdrantVectorStore()
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        
        # Reranker (optional)
        self.reranker = None
        if use_reranking:
            try:
                self.reranker = CohereReranker()
            except Exception as e:
                logger.warning("reranker_init_failed", error=str(e))
                self.use_reranking = False
        
        logger.info(
            "rag_query_engine_initialized",
            collection_name=collection_name,
            embedding_model=embedding_model,
            use_reranking=use_reranking,
        )
    
    async def close(self):
        """Close all clients."""
        await self.vector_store.close()
        await self.embedding_generator.close()
        
        if self.reranker:
            await self.reranker.close()
        
        logger.info("rag_query_engine_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _embed_query(self, query: str) -> List[float]:
        """
        Embed query text.
        
        Args:
            query: Query text.
            
        Returns:
            Query embedding.
        """
        with track_time("query_embedding_seconds"):
            embedding = await self.embedding_generator.embed_text(query)
        
        return embedding
    
    async def _retrieve_candidates(
        self,
        query_embedding: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidates from vector store.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of candidates.
            filters: Metadata filters.
            
        Returns:
            Retrieved documents.
        """
        with track_time("vector_retrieval_seconds"):
            results = await self.vector_store.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=self.score_threshold,
                filters=filters,
            )
        
        logger.info(
            "candidates_retrieved",
            num_candidates=len(results),
        )
        
        return results
    
    async def _rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using Cohere.
        
        Args:
            query: Query text.
            candidates: Retrieved candidates.
            top_n: Rerank top N.
            
        Returns:
            Reranked documents.
        """
        if not self.use_reranking or not self.reranker:
            return candidates
        
        top_n = top_n or len(candidates)
        
        # Prepare for reranking
        texts = [doc["metadata"].get("text", "") for doc in candidates]
        
        with track_time("reranking_seconds"):
            reranked = await self.reranker.rerank(
                query=query,
                documents=texts,
                top_n=top_n,
                metadata=[doc["metadata"] for doc in candidates],
            )
        
        # Format results
        reranked_docs = []
        for result in reranked:
            doc = candidates[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            doc["original_rank"] = result.index
            reranked_docs.append(doc)
        
        logger.info(
            "results_reranked",
            num_reranked=len(reranked_docs),
            top_score=reranked_docs[0]["rerank_score"] if reranked_docs else 0.0,
        )
        
        return reranked_docs
    
    async def query(
        self,
        query_text: str,
        top_k: int = 10,
        rerank_top_n: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Execute RAG query.
        
        Args:
            query_text: User query.
            top_k: Number of candidates to retrieve.
            rerank_top_n: Rerank top N (default: top_k // 2).
            filters: Metadata filters for retrieval.
            
        Returns:
            QueryResult with retrieved documents.
        """
        logger.info(
            "executing_rag_query",
            query_length=len(query_text),
            top_k=top_k,
        )
        
        with track_time("rag_query_total_seconds"):
            # 1. Embed query
            query_embedding = await self._embed_query(query_text)
            
            # 2. Retrieve candidates
            candidates = await self._retrieve_candidates(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
            
            num_retrieved = len(candidates)
            
            # 3. Rerank (optional)
            if self.use_reranking:
                rerank_top_n = rerank_top_n or (top_k // 2)
                candidates = await self._rerank_results(
                    query=query_text,
                    candidates=candidates,
                    top_n=rerank_top_n,
                )
            
            num_after_rerank = len(candidates)
        
        logger.info(
            "rag_query_completed",
            num_retrieved=num_retrieved,
            num_after_rerank=num_after_rerank,
        )
        
        return QueryResult(
            query=query_text,
            retrieved_docs=candidates,
            num_docs_retrieved=num_retrieved,
            num_docs_after_rerank=num_after_rerank,
        )
    
    async def query_with_filters(
        self,
        query_text: str,
        doc_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        legal_area: Optional[str] = None,
        date_range: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> QueryResult:
        """
        Query with metadata filters.
        
        Args:
            query_text: User query.
            doc_type: Filter by document type.
            jurisdiction: Filter by jurisdiction.
            legal_area: Filter by legal area.
            date_range: Filter by date ({"gte": "2020-01-01", "lte": "2024-12-31"}).
            **kwargs: Additional filters or query params.
            
        Returns:
            QueryResult.
        """
        # Build filters
        filters = {}
        
        if doc_type:
            filters["doc_type"] = doc_type
        
        if jurisdiction:
            filters["jurisdiction"] = jurisdiction
        
        if legal_area:
            filters["legal_area"] = legal_area
        
        if date_range:
            filters["date"] = date_range
        
        logger.info("query_with_filters", filters=filters)
        
        return await self.query(query_text, filters=filters, **kwargs)


# Convenience function
async def query_documents(
    collection_name: str,
    query_text: str,
    top_k: int = 10,
    **kwargs,
) -> QueryResult:
    """
    Quick document querying.
    
    Args:
        collection_name: Qdrant collection.
        query_text: Query text.
        top_k: Number of results.
        **kwargs: Passed to RAGQueryEngine.query().
        
    Returns:
        QueryResult.
        
    Example:
        >>> result = await query_documents(
        ...     "legal_docs",
        ...     "договор купли-продажи",
        ...     top_k=5,
        ... )
        >>> for doc in result.retrieved_docs:
        ...     print(doc["metadata"]["text"][:100])
    """
    async with RAGQueryEngine(collection_name=collection_name) as engine:
        return await engine.query(query_text, top_k=top_k, **kwargs)
