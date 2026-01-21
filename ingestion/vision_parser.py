"""
Vision Parser - GPT-4V для обработки изображений и сканов документов.

Этот модуль использует GPT-4 Vision для:
- Распознавания текста в изображениях (OCR alternative)
- Анализа таблиц, графиков, диаграмм
- Извлечения печатей, подписей, штампов
- Обработки сканов контрактов

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import base64
import mimetypes
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal

import aiohttp
from openai import AsyncOpenAI
from PIL import Image

from legal_assistant.core import (
    get_logger,
    get_settings,
    LegalAssistantException,
    DocumentParsingError,
    LLMError,
    track_time,
    track_error,
    track_llm_tokens,
)
from legal_assistant.utils.retry import retry_on_llm_error


logger = get_logger(__name__)
settings = get_settings()


class VisionParserError(DocumentParsingError):
    """Vision parsing error."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message, details=details or {})
        self.model = model


class VisionParser:
    """
    Production-ready GPT-4V parser для изображений.
    
    Features:
    - Multimodal parsing (text + images)
    - Automatic image optimization
    - Structured output prompts
    - Retry logic с fallback
    - Token tracking & metrics
    - Cost estimation
    
    Example:
        >>> parser = VisionParser()
        >>> result = await parser.parse_image("scan.jpg")
        >>> print(result["text"])
    """
    
    # Model configuration
    DEFAULT_MODEL = "gpt-4o"  # GPT-4 Omni (vision + text)
    MAX_IMAGE_SIZE = 20 * 1024 * 1024  # 20MB
    MAX_TOKENS = 4096
    
    # Supported image formats
    SUPPORTED_FORMATS = {
        "image/jpeg": [".jpg", ".jpeg"],
        "image/png": [".png"],
        "image/webp": [".webp"],
        "image/gif": [".gif"],
    }
    
    # System prompts
    SYSTEM_PROMPT = """Ты эксперт по анализу юридических документов.
Твоя задача - точно извлечь весь текст из предоставленного изображения.

Особое внимание обращай на:
- Названия сторон договора
- Даты и номера документов
- Суммы и цифры
- Подписи и печати
- Таблицы и структурированные данные

Сохраняй исходное форматирование и структуру текста."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize Vision Parser.
        
        Args:
            api_key: OpenAI API key. If None, uses config.
            model: Model name. Defaults to gpt-4o.
        """
        self.api_key = api_key or settings.llm.openai_api_key.get_secret_value()
        self.model = model or self.DEFAULT_MODEL
        
        # OpenAI client
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        logger.info(
            "vision_parser_initialized",
            model=self.model,
            max_tokens=self.MAX_TOKENS,
        )
    
    async def close(self):
        """Close OpenAI client."""
        await self.client.close()
        logger.info("vision_parser_closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def _validate_image(self, file_path: Path) -> None:
        """
        Validate image file.
        
        Args:
            file_path: Path to image.
            
        Raises:
            DocumentParsingError: If invalid.
        """
        # Existence
        if not file_path.exists():
            raise DocumentParsingError(
                f"Image not found: {file_path}",
                details={"file_path": str(file_path)},
            )
        
        # Size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_IMAGE_SIZE:
            raise DocumentParsingError(
                f"Image too large: {file_size / 1024 / 1024:.2f}MB (max: {self.MAX_IMAGE_SIZE / 1024 / 1024:.2f}MB)",
                details={
                    "file_path": str(file_path),
                    "file_size": file_size,
                    "max_size": self.MAX_IMAGE_SIZE,
                },
            )
        
        # MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type not in self.SUPPORTED_FORMATS:
            supported = ", ".join(
                ext for exts in self.SUPPORTED_FORMATS.values() for ext in exts
            )
            raise DocumentParsingError(
                f"Unsupported image format: {mime_type}. Supported: {supported}",
                details={
                    "file_path": str(file_path),
                    "mime_type": mime_type,
                },
            )
        
        logger.debug(
            "image_validated",
            file_path=str(file_path),
            file_size=file_size,
            mime_type=mime_type,
        )
    
    def _optimize_image(self, file_path: Path) -> Path:
        """
        Optimize image for API (compress if needed).
        
        Args:
            file_path: Original image path.
            
        Returns:
            Path to optimized image (may be same).
        """
        try:
            img = Image.open(file_path)
            
            # Check if optimization needed
            file_size = file_path.stat().st_size
            if file_size < 5 * 1024 * 1024:  # < 5MB
                return file_path
            
            # Compress
            optimized_path = file_path.with_stem(f"{file_path.stem}_optimized")
            
            # Resize if too large
            max_dimension = 2048
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Save with compression
            img.save(
                optimized_path,
                format=img.format or "JPEG",
                quality=85,
                optimize=True,
            )
            
            logger.info(
                "image_optimized",
                original_size=file_size,
                optimized_size=optimized_path.stat().st_size,
            )
            
            return optimized_path
        
        except Exception as e:
            logger.warning("image_optimization_failed", error=str(e))
            return file_path
    
    def _encode_image(self, file_path: Path) -> str:
        """
        Encode image to base64.
        
        Args:
            file_path: Path to image.
            
        Returns:
            Base64 encoded string.
        """
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    
    @retry_on_llm_error
    async def _call_vision_api(
        self,
        image_base64: str,
        prompt: str,
        max_tokens: int = MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Call GPT-4V API.
        
        Args:
            image_base64: Base64 encoded image.
            prompt: User prompt.
            max_tokens: Max output tokens.
            
        Returns:
            API response with text + usage.
            
        Raises:
            LLMError: If API call fails.
        """
        try:
            with track_time("vision_api_call_seconds"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": self.SYSTEM_PROMPT,
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_base64}",
                                    },
                                },
                            ],
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=0.0,  # Deterministic
                )
                
                # Extract result
                text = response.choices[0].message.content
                usage = response.usage
                
                # Track metrics
                track_llm_tokens(
                    provider="openai",
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                )
                
                logger.info(
                    "vision_api_success",
                    model=self.model,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                )
                
                return {
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.total_tokens,
                    },
                }
        
        except Exception as e:
            track_error("vision_api_call")
            raise LLMError(
                f"Vision API error: {e}",
                provider="openai",
                model=self.model,
                details={"error": str(e)},
            ) from e
    
    async def parse_image(
        self,
        file_path: str | Path,
        *,
        task: Literal["ocr", "table", "signature", "general"] = "general",
        custom_prompt: Optional[str] = None,
        optimize: bool = True,
    ) -> Dict[str, Any]:
        """
        Parse image using GPT-4V.
        
        Args:
            file_path: Path to image file.
            task: Parsing task type:
                - "ocr": Extract all text
                - "table": Extract table structure
                - "signature": Detect signatures/seals
                - "general": General document analysis
            custom_prompt: Override default prompt.
            optimize: Optimize image before upload.
            
        Returns:
            Parsing result:
            {
                "text": "Extracted text",
                "metadata": {
                    "model": "gpt-4o",
                    "task": "ocr",
                    "tokens_used": 1500,
                    ...
                },
            }
            
        Raises:
            DocumentParsingError: If image is invalid.
            LLMError: If API fails.
        """
        file_path = Path(file_path)
        
        # Validate
        self._validate_image(file_path)
        
        # Optimize if needed
        if optimize:
            file_path = self._optimize_image(file_path)
        
        # Encode
        image_base64 = self._encode_image(file_path)
        
        # Build prompt based on task
        if custom_prompt:
            prompt = custom_prompt
        elif task == "ocr":
            prompt = "Извлеки весь текст из этого изображения. Сохраняй форматирование и структуру."
        elif task == "table":
            prompt = "Извлеки данные из таблицы в этом изображении в виде структурированного текста."
        elif task == "signature":
            prompt = "Проанализируй это изображение на наличие подписей, печатей и штампов. Опиши их расположение."
        else:  # general
            prompt = "Проанализируй это юридическое изображение и извлеки всю важную информацию."
        
        logger.info(
            "starting_vision_parse",
            file_path=str(file_path),
            task=task,
            image_size=file_path.stat().st_size,
        )
        
        # Parse
        result = await self._call_vision_api(image_base64, prompt)
        
        # Add metadata
        result["metadata"] = {
            "model": self.model,
            "task": task,
            "tokens_used": result["usage"]["total_tokens"],
            "image_path": str(file_path),
            "image_size": file_path.stat().st_size,
        }
        
        logger.info(
            "vision_parse_completed",
            file_path=str(file_path),
            text_length=len(result["text"]),
            tokens=result["usage"]["total_tokens"],
        )
        
        return result
    
    async def parse_multiple_images(
        self,
        file_paths: List[str | Path],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Parse multiple images in parallel.
        
        Args:
            file_paths: List of image paths.
            **kwargs: Passed to parse_image().
            
        Returns:
            List of parsing results.
        """
        import asyncio
        
        tasks = [
            self.parse_image(fp, **kwargs)
            for fp in file_paths
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "image_parse_failed",
                    file_path=str(file_paths[i]),
                    error=str(result),
                )
                final_results.append({
                    "text": "",
                    "error": str(result),
                    "metadata": {"file_path": str(file_paths[i])},
                })
            else:
                final_results.append(result)
        
        return final_results


# Convenience function
async def parse_image(
    file_path: str | Path,
    **kwargs,
) -> Dict[str, Any]:
    """
    Quick image parsing.
    
    Args:
        file_path: Path to image.
        **kwargs: Passed to VisionParser.parse_image().
        
    Returns:
        Parsing result.
        
    Example:
        >>> result = await parse_image("scan.jpg", task="ocr")
        >>> print(result["text"])
    """
    async with VisionParser() as parser:
        return await parser.parse_image(file_path, **kwargs)
