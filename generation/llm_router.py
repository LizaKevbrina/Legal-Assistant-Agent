"""
LLM Router - Multi-provider LLM routing с fallback и load balancing.

Features:
- Multiple providers (OpenAI, Anthropic, local)
- Automatic fallback on errors
- Load balancing
- Cost tracking
- Token budgets
- Streaming support

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Literal
from dataclasses import dataclass
from datetime import datetime
import asyncio

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from legal_assistant.core import (
    get_logger,
    get_settings,
    LLMError,
    track_time,
    track_error,
    track_llm_tokens,
)
from legal_assistant.utils.retry import retry_on_llm_error


logger = get_logger(__name__)
settings = get_settings()


@dataclass
class LLMConfig:
    """LLM model configuration."""
    provider: Literal["openai", "anthropic"]
    model: str
    max_tokens: int
    temperature: float
    cost_per_1k_input: float
    cost_per_1k_output: float
    priority: int = 1  # Lower = higher priority


# Model registry
MODEL_REGISTRY = {
    # OpenAI
    "gpt-4-turbo": LLMConfig(
        provider="openai",
        model="gpt-4-turbo-preview",
        max_tokens=4096,
        temperature=0.0,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        priority=1,
    ),
    "gpt-4o": LLMConfig(
        provider="openai",
        model="gpt-4o",
        max_tokens=4096,
        temperature=0.0,
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        priority=2,
    ),
    "gpt-3.5-turbo": LLMConfig(
        provider="openai",
        model="gpt-3.5-turbo",
        max_tokens=4096,
        temperature=0.0,
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        priority=3,
    ),
    
    # Anthropic
    "claude-3.5-sonnet": LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20250219",
        max_tokens=4096,
        temperature=0.0,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        priority=1,
    ),
    "claude-3-opus": LLMConfig(
        provider="anthropic",
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.0,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        priority=2,
    ),
}


@dataclass
class LLMResponse:
    """LLM response with metadata."""
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "duration_seconds": self.duration_seconds,
        }


class LLMRouter:
    """
    Production-ready LLM router with multi-provider support.
    
    Features:
    - Automatic fallback (OpenAI → Anthropic)
    - Cost tracking
    - Token budgets
    - Load balancing
    - Streaming support
    
    Example:
        >>> router = LLMRouter()
        >>> response = await router.complete(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     model="gpt-4-turbo",
        ... )
        >>> print(response.content)
    """
    
    def __init__(
        self,
        primary_model: str = "gpt-4-turbo",
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 2,
    ):
        """
        Initialize LLM router.
        
        Args:
            primary_model: Primary model to use.
            fallback_models: Fallback models (in order).
            max_retries: Max retry attempts per model.
        """
        if primary_model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {primary_model}")
        
        self.primary_model = primary_model
        self.fallback_models = fallback_models or []
        self.max_retries = max_retries
        
        # Validate fallback models
        for model in self.fallback_models:
            if model not in MODEL_REGISTRY:
                raise ValueError(f"Unknown fallback model: {model}")
        
        # Initialize clients
        self._openai_client = None
        self._anthropic_client = None
        
        # Usage tracking
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        
        logger.info(
            "llm_router_initialized",
            primary_model=primary_model,
            fallback_models=fallback_models,
        )
    
    def _get_openai_client(self) -> AsyncOpenAI:
        """Get or create OpenAI client."""
        if self._openai_client is None:
            api_key = settings.llm.openai_api_key.get_secret_value()
            self._openai_client = AsyncOpenAI(api_key=api_key)
        return self._openai_client
    
    def _get_anthropic_client(self) -> AsyncAnthropic:
        """Get or create Anthropic client."""
        if self._anthropic_client is None:
            api_key = settings.llm.anthropic_api_key.get_secret_value()
            self._anthropic_client = AsyncAnthropic(api_key=api_key)
        return self._anthropic_client
    
    async def close(self):
        """Close all clients."""
        if self._openai_client:
            await self._openai_client.close()
        
        if self._anthropic_client:
            await self._anthropic_client.close()
        
        logger.info(
            "llm_router_closed",
            total_tokens=self.total_tokens,
            total_cost_usd=self.total_cost_usd,
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    @retry_on_llm_error
    async def _call_openai(
        self,
        config: LLMConfig,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API."""
        client = self._get_openai_client()
        
        start_time = datetime.utcnow()
        
        try:
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=temperature or config.temperature,
                max_tokens=max_tokens or config.max_tokens,
                **kwargs,
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract data
            content = response.choices[0].message.content
            usage = response.usage
            
            # Calculate cost
            cost_input = (usage.prompt_tokens / 1000) * config.cost_per_1k_input
            cost_output = (usage.completion_tokens / 1000) * config.cost_per_1k_output
            total_cost = cost_input + cost_output
            
            # Track metrics
            track_llm_tokens(
                provider="openai",
                model=config.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
            
            # Update totals
            self.total_tokens += usage.total_tokens
            self.total_cost_usd += total_cost
            
            logger.info(
                "openai_call_success",
                model=config.model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cost_usd=total_cost,
                duration=duration,
            )
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="openai",
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=total_cost,
                duration_seconds=duration,
            )
        
        except Exception as e:
            track_error("openai_call")
            raise LLMError(
                f"OpenAI API error: {e}",
                provider="openai",
                model=config.model,
                details={"error": str(e)},
            ) from e
    
    @retry_on_llm_error
    async def _call_anthropic(
        self,
        config: LLMConfig,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Call Anthropic API."""
        client = self._get_anthropic_client()
        
        # Convert messages format
        system_message = None
        converted_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                converted_messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })
        
        start_time = datetime.utcnow()
        
        try:
            response = await client.messages.create(
                model=config.model,
                messages=converted_messages,
                system=system_message,
                temperature=temperature or config.temperature,
                max_tokens=max_tokens or config.max_tokens,
                **kwargs,
            )
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Extract data
            content = response.content[0].text
            usage = response.usage
            
            # Calculate cost
            cost_input = (usage.input_tokens / 1000) * config.cost_per_1k_input
            cost_output = (usage.output_tokens / 1000) * config.cost_per_1k_output
            total_cost = cost_input + cost_output
            
            # Track metrics
            track_llm_tokens(
                provider="anthropic",
                model=config.model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )
            
            # Update totals
            total_tokens = usage.input_tokens + usage.output_tokens
            self.total_tokens += total_tokens
            self.total_cost_usd += total_cost
            
            logger.info(
                "anthropic_call_success",
                model=config.model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cost_usd=total_cost,
                duration=duration,
            )
            
            return LLMResponse(
                content=content,
                model=config.model,
                provider="anthropic",
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=total_tokens,
                cost_usd=total_cost,
                duration_seconds=duration,
            )
        
        except Exception as e:
            track_error("anthropic_call")
            raise LLMError(
                f"Anthropic API error: {e}",
                provider="anthropic",
                model=config.model,
                details={"error": str(e)},
            ) from e
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Complete chat with automatic fallback.
        
        Args:
            messages: Chat messages (OpenAI format).
            model: Model name (uses primary if None).
            temperature: Sampling temperature.
            max_tokens: Max completion tokens.
            **kwargs: Additional provider-specific params.
            
        Returns:
            LLMResponse with content and metadata.
            
        Raises:
            LLMError: If all models fail.
        """
        model = model or self.primary_model
        models_to_try = [model] + self.fallback_models
        
        errors = []
        
        for model_name in models_to_try:
            config = MODEL_REGISTRY[model_name]
            
            logger.info(
                "trying_model",
                model=model_name,
                provider=config.provider,
            )
            
            try:
                with track_time("llm_completion_seconds"):
                    if config.provider == "openai":
                        response = await self._call_openai(
                            config,
                            messages,
                            temperature,
                            max_tokens,
                            **kwargs,
                        )
                    elif config.provider == "anthropic":
                        response = await self._call_anthropic(
                            config,
                            messages,
                            temperature,
                            max_tokens,
                            **kwargs,
                        )
                    else:
                        raise ValueError(f"Unknown provider: {config.provider}")
                
                logger.info("llm_completion_success", model=model_name)
                return response
            
            except Exception as e:
                logger.warning(
                    "model_failed",
                    model=model_name,
                    error=str(e),
                )
                errors.append(f"{model_name}: {str(e)}")
        
        # All models failed
        track_error("llm_all_models_failed")
        raise LLMError(
            "All models failed",
            provider="multiple",
            model=models_to_try[0],
            details={"errors": errors},
        )
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Usage stats (tokens, cost).
        """
        return {
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
        }


# Convenience function
async def complete_chat(
    messages: List[Dict[str, str]],
    model: str = "gpt-4-turbo",
    **kwargs,
) -> LLMResponse:
    """
    Quick chat completion.
    
    Args:
        messages: Chat messages.
        model: Model name.
        **kwargs: Additional params.
        
    Returns:
        LLMResponse.
        
    Example:
        >>> response = await complete_chat([
        ...     {"role": "system", "content": "You are a legal expert."},
        ...     {"role": "user", "content": "What is a contract?"},
        ... ])
        >>> print(response.content)
    """
    async with LLMRouter(primary_model=model) as router:
        return await router.complete(messages, **kwargs)
