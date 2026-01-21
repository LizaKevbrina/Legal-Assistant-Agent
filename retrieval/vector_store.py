"""
Vector Store - Qdrant client для хранения и поиска векторов.

Features:
- Collection management (create, delete, info)
- Vector upsert с metadata
- Similarity search (dense vectors)
- Metadata filtering
- Batch operations
- Connection pooling
- Error handling + retry

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import asyncio

from qdrant_client import QdrantClient
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchRequest,
    CollectionInfo,
    UpdateStatus,
)

from legal_assistant.core import (
    get_logger,
    get_settings,
    VectorStoreError,
    track_time,
    track_error,
)
from legal_assistant.utils.retry import retry_on_vector_store_error


logger = get_logger(__name__)
settings = get_settings()


class QdrantVectorStore:
    """
    Production-ready Qdrant vector store client.
    
    Features:
    - Async operations
    - Batch upsert
    - Metadata filtering
    - HNSW indexing
    - Connection pooling
    - Error handling
    
    Example:
        >>> store = QdrantVectorStore()
        >>> await store.create_collection("docs", vector_size=3072)
        >>> await store.upsert_vectors("docs", vectors, metadata)
        >>> results = await store.search("docs", query_vector, top_k=5)
    """
    
    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 60,
        prefer_grpc: bool = True,
    ):
        """
        Initialize Qdrant client.
        
        Args:
            url: Qdrant server URL. If None, uses config.
            api_key: API key for Qdrant Cloud.
            timeout: Request timeout (seconds).
            prefer_grpc: Use gRPC for better performance.
        """
        self.url = url or settings.qdrant.url
        self.api_key = api_key or (
            settings.qdrant.api_key.get_secret_value()
            if settings.qdrant.api_key
            else None
        )
        
        # Async client
        self.client = AsyncQdrantClient(
            url=self.url,
            api_key=self.api_key,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )
        
        logger.info(
            "qdrant_client_initialized",
            url=self.url,
            timeout=timeout,
            prefer_grpc=prefer_grpc,
        )
    
    async def close(self):
        """Close Qdrant client."""
        await self.client.close()
        logger.info("qdrant_client_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @retry_on_vector_store_error
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        on_disk: bool = False,
        hnsw_config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create vector collection.
        
        Args:
            collection_name: Collection name.
            vector_size: Vector dimension.
            distance: Distance metric (COSINE, EUCLID, DOT).
            on_disk: Store vectors on disk (for large datasets).
            hnsw_config: HNSW index params (m, ef_construct).
            
        Returns:
            True if created successfully.
            
        Raises:
            VectorStoreError: If creation fails.
        """
        try:
            # Check if exists
            exists = await self.client.collection_exists(collection_name)
            if exists:
                logger.warning(
                    "collection_already_exists",
                    collection_name=collection_name,
                )
                return False
            
            # Default HNSW config
            if hnsw_config is None:
                hnsw_config = {
                    "m": 16,  # Number of edges per node
                    "ef_construct": 200,  # Construction time accuracy
                }
            
            # Create
            logger.info(
                "creating_collection",
                collection_name=collection_name,
                vector_size=vector_size,
                distance=distance.value,
            )
            
            with track_time("collection_creation_seconds"):
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance,
                        on_disk=on_disk,
                    ),
                    hnsw_config=hnsw_config,
                )
            
            logger.info("collection_created", collection_name=collection_name)
            return True
        
        except Exception as e:
            track_error("collection_creation")
            raise VectorStoreError(
                f"Failed to create collection: {e}",
                operation="create_collection",
                details={
                    "collection_name": collection_name,
                    "error": str(e),
                },
            ) from e
    
    @retry_on_vector_store_error
    async def delete_collection(self, collection_name: str) -> bool:
        """
        Delete collection.
        
        Args:
            collection_name: Collection name.
            
        Returns:
            True if deleted.
        """
        try:
            logger.info("deleting_collection", collection_name=collection_name)
            
            await self.client.delete_collection(collection_name)
            
            logger.info("collection_deleted", collection_name=collection_name)
            return True
        
        except Exception as e:
            track_error("collection_deletion")
            raise VectorStoreError(
                f"Failed to delete collection: {e}",
                operation="delete_collection",
                details={"collection_name": collection_name, "error": str(e)},
            ) from e
    
    @retry_on_vector_store_error
    async def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get collection info.
        
        Args:
            collection_name: Collection name.
            
        Returns:
            Collection metadata.
        """
        try:
            info = await self.client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status.value,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.value,
                },
            }
        
        except Exception as e:
            raise VectorStoreError(
                f"Failed to get collection info: {e}",
                operation="get_collection_info",
                details={"collection_name": collection_name, "error": str(e)},
            ) from e
    
    @retry_on_vector_store_error
    async def upsert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
    ) -> int:
        """
        Upsert vectors with metadata.
        
        Args:
            collection_name: Collection name.
            vectors: List of vectors.
            metadata: List of metadata dicts (same length as vectors).
            ids: Optional point IDs (generated if None).
            batch_size: Batch size for upserting.
            
        Returns:
            Number of vectors upserted.
            
        Raises:
            VectorStoreError: If upsert fails.
        """
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]
        
        try:
            logger.info(
                "upserting_vectors",
                collection_name=collection_name,
                num_vectors=len(vectors),
            )
            
            # Prepare points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=meta,
                )
                for point_id, vector, meta in zip(ids, vectors, metadata)
            ]
            
            # Batch upsert
            total_upserted = 0
            with track_time("vector_upsert_seconds"):
                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    
                    result = await self.client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True,
                    )
                    
                    if result.status == UpdateStatus.COMPLETED:
                        total_upserted += len(batch)
                    
                    logger.debug(
                        "batch_upserted",
                        batch_num=i // batch_size + 1,
                        batch_size=len(batch),
                    )
            
            logger.info(
                "vectors_upserted",
                collection_name=collection_name,
                num_vectors=total_upserted,
            )
            
            return total_upserted
        
        except Exception as e:
            track_error("vector_upsert")
            raise VectorStoreError(
                f"Failed to upsert vectors: {e}",
                operation="upsert_vectors",
                details={
                    "collection_name": collection_name,
                    "num_vectors": len(vectors),
                    "error": str(e),
                },
            ) from e
    
    @retry_on_vector_store_error
    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search similar vectors.
        
        Args:
            collection_name: Collection name.
            query_vector: Query vector.
            top_k: Number of results.
            score_threshold: Minimum similarity score.
            filters: Metadata filters (e.g., {"doc_type": "contract"}).
            
        Returns:
            List of search results with scores and metadata.
        """
        try:
            logger.debug(
                "searching_vectors",
                collection_name=collection_name,
                top_k=top_k,
            )
            
            # Build filters
            qdrant_filter = None
            if filters:
                conditions = []
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Multiple values (OR)
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(any=value),
                            )
                        )
                    elif isinstance(value, dict) and ("gte" in value or "lte" in value):
                        # Range filter
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(
                                    gte=value.get("gte"),
                                    lte=value.get("lte"),
                                ),
                            )
                        )
                    else:
                        # Single value
                        conditions.append(
                            FieldCondition(
                                key=key,
                                match=MatchValue(value=value),
                            )
                        )
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
            
            # Search
            with track_time("vector_search_seconds"):
                results = await self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=qdrant_filter,
                    with_payload=True,
                    with_vectors=False,
                )
            
            # Format results
            formatted_results = [
                {
                    "id": result.id,
                    "score": result.score,
                    "metadata": result.payload,
                }
                for result in results
            ]
            
            logger.info(
                "search_completed",
                collection_name=collection_name,
                num_results=len(formatted_results),
            )
            
            return formatted_results
        
        except Exception as e:
            track_error("vector_search")
            raise VectorStoreError(
                f"Vector search failed: {e}",
                operation="search",
                details={
                    "collection_name": collection_name,
                    "top_k": top_k,
                    "error": str(e),
                },
            ) from e
    
    @retry_on_vector_store_error
    async def delete_vectors(
        self,
        collection_name: str,
        ids: List[str],
    ) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            collection_name: Collection name.
            ids: List of point IDs to delete.
            
        Returns:
            Number of vectors deleted.
        """
        try:
            logger.info(
                "deleting_vectors",
                collection_name=collection_name,
                num_ids=len(ids),
            )
            
            result = await self.client.delete(
                collection_name=collection_name,
                points_selector=ids,
                wait=True,
            )
            
            if result.status == UpdateStatus.COMPLETED:
                logger.info("vectors_deleted", num_deleted=len(ids))
                return len(ids)
            
            return 0
        
        except Exception as e:
            track_error("vector_deletion")
            raise VectorStoreError(
                f"Failed to delete vectors: {e}",
                operation="delete_vectors",
                details={
                    "collection_name": collection_name,
                    "num_ids": len(ids),
                    "error": str(e),
                },
            ) from e
    
    async def health_check(self) -> bool:
        """
        Check if Qdrant is healthy.
        
        Returns:
            True if healthy.
        """
        try:
            collections = await self.client.get_collections()
            logger.debug("qdrant_health_check", status="healthy")
            return True
        except Exception as e:
            logger.error("qdrant_health_check", status="unhealthy", error=str(e))
            return False


# Convenience functions
async def create_collection(
    collection_name: str,
    vector_size: int,
    **kwargs,
) -> bool:
    """Quick collection creation."""
    async with QdrantVectorStore() as store:
        return await store.create_collection(collection_name, vector_size, **kwargs)


async def upsert_vectors(
    collection_name: str,
    vectors: List[List[float]],
    metadata: List[Dict[str, Any]],
    **kwargs,
) -> int:
    """Quick vector upsert."""
    async with QdrantVectorStore() as store:
        return await store.upsert_vectors(collection_name, vectors, metadata, **kwargs)


async def search_vectors(
    collection_name: str,
    query_vector: List[float],
    top_k: int = 10,
    **kwargs,
) -> List[Dict[str, Any]]:
    """Quick vector search."""
    async with QdrantVectorStore() as store:
        return await store.search(collection_name, query_vector, top_k, **kwargs)
