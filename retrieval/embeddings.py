"""
Embeddings - OpenAI embeddings с кэшированием и батчингом.

Features:
- Batch processing (эффективная генерация)
- Redis caching (reduce API calls)
- Multiple models support
- Token counting & cost tracking
- Error handling + retry
- Rate limiting

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import hashlib
from typing import List, Optional, Dict, Any
import asyncio

from openai import AsyncOpenAI
import tiktoken

from legal_assistant.core import (
    get_logger,
    get_settings,
    LLMError,
    track_time,
    track_error,
    track_llm_tokens,
)
from legal_assistant.utils.retry import retry_on_embedding_error


logger = get_logger(__name__)
settings = get_settings()


class EmbeddingGenerator:
    """
    Production-ready embedding generator.
    
    Features:
    - Batch processing (до 2048 текстов за раз)
    - Redis caching
    - Token counting
    - Cost tracking
    - Error handling
    
    Example:
        >>> generator = EmbeddingGenerator()
        >>> embeddings = await generator.embed_texts(["text1", "text2"])
        >>> print(len(embeddings))  # 2
        >>> print(len(embeddings[0]))  # 3072
    """
    
    # Model configurations
    MODELS = {
        "text-embedding-3-large": {
            "dimensions": 3072,
            "max_tokens": 8191,
            "cost_per_1k": 0.00013,  # USD
        },
        "text-embedding-3-small": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k": 0.00002,
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "max_tokens": 8191,
            "cost_per_1k": 0.0001,
        },
    }
    
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
        batch_size: int = 100,
        use_cache: bool = True,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: OpenAI embedding model name.
            api_key: OpenAI API key.
            batch_size: Batch size for embedding generation.
            use_cache: Enable Redis caching.
        """
        if model not in self.MODELS:
            raise ValueError(
                f"Unknown model: {model}. "
                f"Available: {list(self.MODELS.keys())}"
            )
        
        self.model = model
        self.model_config = self.MODELS[model]
        self.batch_size = batch_size
        self.use_cache = use_cache
        
        # OpenAI client
        api_key = api_key or settings.llm.openai_api_key.get_secret_value()
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Tokenizer
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        
        # Cache
        self._cache = None
        if use_cache:
            try:
                import redis.asyncio as aioredis
                self._cache = aioredis.from_url(
                    settings.redis.url,
                    encoding="utf-8",
                    decode_responses=False,  # Store binary
                )
            except ImportError:
                logger.warning("Redis not available, caching disabled")
                self.use_cache = False
        
        logger.info(
            "embedding_generator_initialized",
            model=model,
            dimensions=self.model_config["dimensions"],
            batch_size=batch_size,
            use_cache=use_cache,
        )
    
    async def close(self):
        """Close clients."""
        await self.client.close()
        
        if self._cache:
            await self._cache.close()
        
        logger.info("embedding_generator_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text.
            
        Returns:
            Token count.
        """
        return len(self.tokenizer.encode(text))
    
    def _truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """
        Truncate text to max tokens.
        
        Args:
            text: Input text.
            max_tokens: Max tokens (uses model max if None).
            
        Returns:
            Truncated text.
        """
        max_tokens = max_tokens or self.model_config["max_tokens"]
        tokens = self.tokenizer.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate
        truncated_tokens = tokens[:max_tokens]
        truncated_text = self.tokenizer.decode(truncated_tokens)
        
        logger.warning(
            "text_truncated",
            original_tokens=len(tokens),
            truncated_tokens=len(truncated_tokens),
        )
        
        return truncated_text
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text.
            
        Returns:
            Cache key (hex string).
        """
        # Hash text + model name
        content = f"{self.model}:{text}"
        key_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"embedding:{key_hash}"
    
    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding.
        
        Args:
            text: Input text.
            
        Returns:
            Cached embedding or None.
        """
        if not self.use_cache or not self._cache:
            return None
        
        try:
            cache_key = self._get_cache_key(text)
            cached = await self._cache.get(cache_key)
            
            if cached:
                # Deserialize
                import pickle
                embedding = pickle.loads(cached)
                logger.debug("embedding_cache_hit", cache_key=cache_key)
                return embedding
        
        except Exception as e:
            logger.warning("cache_get_error", error=str(e))
        
        return None
    
    async def _set_cached_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: int = 86400 * 7,  # 7 days
    ) -> None:
        """
        Cache embedding.
        
        Args:
            text: Input text.
            embedding: Embedding vector.
            ttl: Time to live (seconds).
        """
        if not self.use_cache or not self._cache:
            return
        
        try:
            import pickle
            cache_key = self._get_cache_key(text)
            serialized = pickle.dumps(embedding)
            
            await self._cache.setex(cache_key, ttl, serialized)
            logger.debug("embedding_cached", cache_key=cache_key)
        
        except Exception as e:
            logger.warning("cache_set_error", error=str(e))
    
    @retry_on_embedding_error
    async def _call_embedding_api(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Call OpenAI embedding API.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
            
        Raises:
            LLMError: If API call fails.
        """
        try:
            with track_time("embedding_api_call_seconds"):
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
            
            # Extract embeddings (preserve order)
            embeddings = [item.embedding for item in response.data]
            
            # Track metrics
            total_tokens = response.usage.total_tokens
            cost = (total_tokens / 1000) * self.model_config["cost_per_1k"]
            
            track_llm_tokens(
                provider="openai",
                model=self.model,
                prompt_tokens=total_tokens,
                completion_tokens=0,
            )
            
            logger.info(
                "embedding_api_success",
                model=self.model,
                num_texts=len(texts),
                total_tokens=total_tokens,
                cost_usd=cost,
            )
            
            return embeddings
        
        except Exception as e:
            track_error("embedding_api_call")
            raise LLMError(
                f"Embedding API error: {e}",
                provider="openai",
                model=self.model,
                details={
                    "num_texts": len(texts),
                    "error": str(e),
                },
            ) from e
    
    async def embed_text(self, text: str) -> List[float]:
        """
        Embed single text.
        
        Args:
            text: Input text.
            
        Returns:
            Embedding vector.
        """
        # Check cache
        cached = await self._get_cached_embedding(text)
        if cached:
            return cached
        
        # Truncate if needed
        text = self._truncate_text(text)
        
        # Generate
        embeddings = await self._call_embedding_api([text])
        embedding = embeddings[0]
        
        # Cache
        await self._set_cached_embedding(text, embedding)
        
        return embedding
    
    async def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Embed multiple texts with batching.
        
        Args:
            texts: List of texts.
            show_progress: Show progress logs.
            
        Returns:
            List of embedding vectors (same order as input).
        """
        if not texts:
            return []
        
        logger.info("embedding_texts", num_texts=len(texts))
        
        embeddings = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = await self._get_cached_embedding(text)
            if cached:
                embeddings[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(self._truncate_text(text))
        
        logger.info(
            "cache_stats",
            total=len(texts),
            cached=len(texts) - len(uncached_texts),
            uncached=len(uncached_texts),
        )
        
        # Batch process uncached texts
        if uncached_texts:
            all_new_embeddings = []
            
            for batch_start in range(0, len(uncached_texts), self.batch_size):
                batch = uncached_texts[batch_start:batch_start + self.batch_size]
                
                if show_progress:
                    logger.info(
                        "processing_batch",
                        batch_num=batch_start // self.batch_size + 1,
                        batch_size=len(batch),
                    )
                
                batch_embeddings = await self._call_embedding_api(batch)
                all_new_embeddings.extend(batch_embeddings)
            
            # Place embeddings in correct positions
            for idx, embedding in zip(uncached_indices, all_new_embeddings):
                embeddings[idx] = embedding
                
                # Cache
                await self._set_cached_embedding(texts[idx], embedding)
        
        logger.info("embedding_completed", num_texts=len(texts))
        
        return embeddings
    
    def get_dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.model_config["dimensions"]
    
    def estimate_cost(self, num_tokens: int) -> float:
        """
        Estimate cost for embedding.
        
        Args:
            num_tokens: Number of tokens.
            
        Returns:
            Estimated cost in USD.
        """
        return (num_tokens / 1000) * self.model_config["cost_per_1k"]


# Convenience functions
async def embed_text(text: str, model: str = "text-embedding-3-large") -> List[float]:
    """Quick single text embedding."""
    async with EmbeddingGenerator(model=model) as generator:
        return await generator.embed_text(text)


async def embed_texts(
    texts: List[str],
    model: str = "text-embedding-3-large",
    **kwargs,
) -> List[List[float]]:
    """Quick batch text embedding."""
    async with EmbeddingGenerator(model=model) as generator:
        return await generator.embed_texts(texts, **kwargs)
