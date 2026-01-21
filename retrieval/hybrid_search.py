"""
Hybrid Search - Комбинация dense (vector) и sparse (BM25) поиска.

Hybrid search улучшает качество retrieval:
- Dense search: семантическая схожесть (embeddings)
- Sparse search: keyword matching (BM25)
- Fusion: RRF (Reciprocal Rank Fusion)

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
)


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class SearchResult:
    """
    Search result с метаданными.
    
    Attributes:
        id: Document/chunk ID
        score: Similarity score
        text: Document text
        metadata: Additional metadata
        rank: Result rank (1-based)
    """
    id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    rank: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "score": self.score,
            "text": self.text,
            "metadata": self.metadata,
            "rank": self.rank,
        }


class BM25:
    """
    BM25 (Best Matching 25) для keyword-based поиска.
    
    Simplified implementation для in-memory usage.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.
        
        Args:
            k1: Term frequency saturation parameter.
            b: Length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        
        # Will be set during indexing
        self.corpus = []
        self.doc_lengths = []
        self.avg_doc_length = 0.0
        self.idf_scores = {}
        self.num_docs = 0
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization (lowercase + split).
        
        Args:
            text: Input text.
            
        Returns:
            List of tokens.
        """
        return text.lower().split()
    
    def index_documents(self, documents: List[str]) -> None:
        """
        Index documents for BM25.
        
        Args:
            documents: List of document texts.
        """
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.num_docs = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
        
        # Calculate IDF scores
        df = Counter()  # Document frequency
        for doc in self.corpus:
            unique_tokens = set(doc)
            for token in unique_tokens:
                df[token] += 1
        
        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        self.idf_scores = {
            token: math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in df.items()
        }
        
        logger.info(
            "bm25_indexed",
            num_docs=self.num_docs,
            unique_terms=len(self.idf_scores),
        )
    
    def get_scores(self, query: str) -> List[float]:
        """
        Get BM25 scores for query against all documents.
        
        Args:
            query: Query text.
            
        Returns:
            List of scores (one per document).
        """
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx, doc in enumerate(self.corpus):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]
            
            # Count term frequencies in doc
            tf = Counter(doc)
            
            for token in query_tokens:
                if token not in self.idf_scores:
                    continue
                
                # Term frequency
                term_freq = tf.get(token, 0)
                
                # IDF score
                idf = self.idf_scores[token]
                
                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                
                score += idf * (numerator / denominator)
            
            scores.append(score)
        
        return scores
    
    def search(self, query: str, top_k: int = 10) -> List[int]:
        """
        Search documents and return top-k indices.
        
        Args:
            query: Query text.
            top_k: Number of results.
            
        Returns:
            List of document indices sorted by score.
        """
        scores = self.get_scores(query)
        
        # Sort by score (descending)
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        # Return top-k indices
        return [idx for idx, score in ranked[:top_k]]


class HybridSearch:
    """
    Hybrid search combining dense (vector) and sparse (BM25) retrieval.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    
    Example:
        >>> search = HybridSearch()
        >>> search.index_documents(texts, embeddings, metadata)
        >>> results = await search.search(
        ...     query_text="договор купли-продажи",
        ...     query_embedding=embedding,
        ...     top_k=10,
        ... )
    """
    
    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
    ):
        """
        Initialize hybrid search.
        
        Args:
            dense_weight: Weight for dense (vector) search.
            sparse_weight: Weight for sparse (BM25) search.
            rrf_k: RRF constant (typical: 60).
        """
        if not math.isclose(dense_weight + sparse_weight, 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        
        # BM25 index
        self.bm25 = BM25()
        
        # Storage
        self.texts = []
        self.embeddings = []
        self.metadata = []
        self.indexed = False
        
        logger.info(
            "hybrid_search_initialized",
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
            rrf_k=rrf_k,
        )
    
    def index_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Index documents for hybrid search.
        
        Args:
            texts: Document texts.
            embeddings: Document embeddings.
            metadata: Document metadata.
        """
        if not (len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("texts, embeddings, metadata must have same length")
        
        logger.info("indexing_documents", num_docs=len(texts))
        
        self.texts = texts
        self.embeddings = embeddings
        self.metadata = metadata
        
        # Index BM25
        self.bm25.index_documents(texts)
        
        self.indexed = True
        logger.info("documents_indexed", num_docs=len(texts))
    
    def _dense_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Dense (vector) search using cosine similarity.
        
        Args:
            query_embedding: Query vector.
            top_k: Number of results.
            
        Returns:
            List of (index, score) tuples.
        """
        if not self.embeddings:
            return []
        
        # Cosine similarity
        scores = []
        for doc_embedding in self.embeddings:
            # Dot product
            dot_product = sum(a * b for a, b in zip(query_embedding, doc_embedding))
            
            # Norms
            query_norm = math.sqrt(sum(x * x for x in query_embedding))
            doc_norm = math.sqrt(sum(x * x for x in doc_embedding))
            
            # Cosine similarity
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0.0
            
            scores.append(similarity)
        
        # Sort by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return ranked[:top_k]
    
    def _sparse_search(
        self,
        query_text: str,
        top_k: int,
    ) -> List[Tuple[int, float]]:
        """
        Sparse (BM25) search.
        
        Args:
            query_text: Query text.
            top_k: Number of results.
            
        Returns:
            List of (index, score) tuples.
        """
        scores = self.bm25.get_scores(query_text)
        
        # Sort by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return ranked[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[int, float]],
        sparse_results: List[Tuple[int, float]],
    ) -> List[int]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(weight / (k + rank))
        
        Args:
            dense_results: Dense search results (index, score).
            sparse_results: Sparse search results (index, score).
            
        Returns:
            List of document indices sorted by fused score.
        """
        # Calculate RRF scores
        rrf_scores = {}
        
        # Dense results
        for rank, (idx, score) in enumerate(dense_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                self.dense_weight / (self.rrf_k + rank)
            )
        
        # Sparse results
        for rank, (idx, score) in enumerate(sparse_results, start=1):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (
                self.sparse_weight / (self.rrf_k + rank)
            )
        
        # Sort by RRF score
        ranked = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        
        return [idx for idx, score in ranked]
    
    def search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 10,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search combining dense + sparse.
        
        Args:
            query_text: Query text for BM25.
            query_embedding: Query embedding for vector search.
            top_k: Final number of results.
            dense_top_k: Candidates from dense search (default: top_k * 2).
            sparse_top_k: Candidates from sparse search (default: top_k * 2).
            
        Returns:
            List of SearchResult objects.
        """
        if not self.indexed:
            raise RuntimeError("Documents not indexed. Call index_documents() first.")
        
        dense_top_k = dense_top_k or (top_k * 2)
        sparse_top_k = sparse_top_k or (top_k * 2)
        
        logger.debug(
            "hybrid_search_started",
            top_k=top_k,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )
        
        with track_time("hybrid_search_seconds"):
            # Dense search
            dense_results = self._dense_search(query_embedding, dense_top_k)
            
            # Sparse search
            sparse_results = self._sparse_search(query_text, sparse_top_k)
            
            # Fusion
            fused_indices = self._reciprocal_rank_fusion(
                dense_results,
                sparse_results,
            )
            
            # Build results
            results = []
            for rank, idx in enumerate(fused_indices[:top_k], start=1):
                # Calculate combined score (for display)
                dense_score = next(
                    (score for i, score in dense_results if i == idx),
                    0.0,
                )
                sparse_score = next(
                    (score for i, score in sparse_results if i == idx),
                    0.0,
                )
                
                combined_score = (
                    self.dense_weight * dense_score +
                    self.sparse_weight * sparse_score
                )
                
                results.append(SearchResult(
                    id=self.metadata[idx].get("id", str(idx)),
                    score=combined_score,
                    text=self.texts[idx],
                    metadata=self.metadata[idx],
                    rank=rank,
                ))
        
        logger.info(
            "hybrid_search_completed",
            num_results=len(results),
            top_score=results[0].score if results else 0.0,
        )
        
        return results
