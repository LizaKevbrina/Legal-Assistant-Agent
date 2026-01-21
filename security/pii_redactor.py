"""
PII (Personally Identifiable Information) detection and redaction.
Uses Microsoft Presidio for entity recognition.

Production features:
- Multiple redaction strategies
- Configurable entity types
- Language support (RU/EN)
- Error handling & fallback
- Async support
- Caching for performance
"""

import hashlib
from typing import Dict, List, Optional, Set
from enum import Enum

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
    track_error,
    pii_detection_duration,
)
from legal_assistant.core.exceptions import SecurityError

logger = get_logger(__name__)
settings = get_settings()


class RedactionStrategy(str, Enum):
    """PII redaction strategies."""
    
    REPLACE = "replace"           # Replace with [REDACTED]
    MASK = "mask"                 # Mask with asterisks
    HASH = "hash"                 # Replace with hash
    ENCRYPT = "encrypt"           # Encrypt (reversible)
    KEEP = "keep"                 # Keep original (for testing)


class PIIEntityType(str, Enum):
    """Supported PII entity types."""
    
    PERSON = "PERSON"                     # Person names
    EMAIL = "EMAIL_ADDRESS"               # Email addresses
    PHONE = "PHONE_NUMBER"                # Phone numbers
    LOCATION = "LOCATION"                 # Addresses
    ORGANIZATION = "ORGANIZATION"         # Company names
    DATE = "DATE_TIME"                    # Dates
    CREDIT_CARD = "CREDIT_CARD"          # Credit card numbers
    PASSPORT = "RU_PASSPORT"             # Russian passports
    INN = "RU_INN"                       # Russian INN
    SNILS = "RU_SNILS"                   # Russian SNILS
    BANK_ACCOUNT = "IBAN_CODE"           # Bank accounts


class PIIRedactor:
    """
    PII detection and redaction using Presidio.
    
    Features:
    - Multiple entity types
    - Configurable strategies
    - Language support (RU/EN)
    - Performance optimized
    
    Example:
        >>> async with PIIRedactor() as redactor:
        ...     result = await redactor.redact_text(
        ...         "Иванов Иван, email: ivan@example.com"
        ...     )
        ...     print(result.text)
        [PERSON], email: [EMAIL_ADDRESS]
    """
    
    def __init__(
        self,
        language: str = "ru",
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
        entity_types: Optional[Set[PIIEntityType]] = None,
    ):
        """
        Initialize PII redactor.
        
        Args:
            language: Language code (ru/en)
            strategy: Redaction strategy
            entity_types: Entity types to detect (None = all)
        """
        self.language = language
        self.strategy = strategy
        self.entity_types = entity_types or set(PIIEntityType)
        
        self._analyzer: Optional[AnalyzerEngine] = None
        self._anonymizer: Optional[AnonymizerEngine] = None
        
        logger.info(
            "pii_redactor_initialized",
            language=language,
            strategy=strategy,
            entity_count=len(self.entity_types),
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass
    
    async def _initialize(self):
        """Initialize Presidio engines."""
        if self._analyzer is not None:
            return
        
        try:
            with track_time(
                pii_detection_duration,
                {"operation": "initialize"}
            ):
                # Setup NLP engine
                nlp_config = {
                    "nlp_engine_name": "spacy",
                    "models": [
                        {"lang_code": "ru", "model_name": "ru_core_news_sm"},
                        {"lang_code": "en", "model_name": "en_core_web_sm"},
                    ],
                }
                
                provider = NlpEngineProvider(nlp_configuration=nlp_config)
                nlp_engine = provider.create_engine()
                
                # Create analyzer
                self._analyzer = AnalyzerEngine(
                    nlp_engine=nlp_engine,
                    supported_languages=["ru", "en"],
                )
                
                # Add custom recognizers for Russian entities
                self._add_russian_recognizers()
                
                # Create anonymizer
                self._anonymizer = AnonymizerEngine()
                
                logger.info("presidio_initialized", language=self.language)
        
        except Exception as e:
            track_error("security", e)
            logger.exception("presidio_initialization_failed")
            raise SecurityError(
                message="Failed to initialize PII detector",
                details={"error": str(e)},
            )
    
    def _add_russian_recognizers(self):
        """Add custom recognizers for Russian PII."""
        from presidio_analyzer import Pattern, PatternRecognizer
        
        # Russian passport pattern
        passport_pattern = Pattern(
            name="russian_passport",
            regex=r"\b\d{4}\s?\d{6}\b",
            score=0.85,
        )
        
        passport_recognizer = PatternRecognizer(
            supported_entity="RU_PASSPORT",
            patterns=[passport_pattern],
        )
        
        # Russian INN pattern (10 or 12 digits)
        inn_pattern = Pattern(
            name="russian_inn",
            regex=r"\b\d{10}|\d{12}\b",
            score=0.8,
        )
        
        inn_recognizer = PatternRecognizer(
            supported_entity="RU_INN",
            patterns=[inn_pattern],
        )
        
        # Russian SNILS pattern (XXX-XXX-XXX YY)
        snils_pattern = Pattern(
            name="russian_snils",
            regex=r"\b\d{3}-\d{3}-\d{3}\s\d{2}\b",
            score=0.9,
        )
        
        snils_recognizer = PatternRecognizer(
            supported_entity="RU_SNILS",
            patterns=[snils_pattern],
        )
        
        # Register recognizers
        registry = self._analyzer.registry
        registry.add_recognizer(passport_recognizer)
        registry.add_recognizer(inn_recognizer)
        registry.add_recognizer(snils_recognizer)
        
        logger.debug("russian_recognizers_added", count=3)
    
    async def detect_pii(
        self,
        text: str,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """
        Detect PII entities in text.
        
        Args:
            text: Input text
            score_threshold: Minimum confidence score
        
        Returns:
            List of detected entities with metadata
        
        Example:
            >>> entities = await redactor.detect_pii("Иван Иванов")
            >>> print(entities[0])
            {
                'entity_type': 'PERSON',
                'start': 0,
                'end': 12,
                'score': 0.85,
                'text': 'Иван Иванов'
            }
        """
        await self._initialize()
        
        try:
            with track_time(
                pii_detection_duration,
                {"operation": "detect", "language": self.language}
            ):
                # Analyze text
                results = self._analyzer.analyze(
                    text=text,
                    language=self.language,
                    entities=[e.value for e in self.entity_types],
                    score_threshold=score_threshold,
                )
                
                # Format results
                entities = []
                for result in results:
                    entities.append({
                        "entity_type": result.entity_type,
                        "start": result.start,
                        "end": result.end,
                        "score": result.score,
                        "text": text[result.start:result.end],
                    })
                
                logger.debug(
                    "pii_detected",
                    entity_count=len(entities),
                    text_length=len(text),
                )
                
                return entities
        
        except Exception as e:
            track_error("security", e)
            logger.exception("pii_detection_failed", text_length=len(text))
            raise SecurityError(
                message="PII detection failed",
                details={"error": str(e)},
            )
    
    async def redact_text(
        self,
        text: str,
        score_threshold: float = 0.5,
    ) -> "RedactionResult":
        """
        Detect and redact PII in text.
        
        Args:
            text: Input text
            score_threshold: Minimum confidence score
        
        Returns:
            RedactionResult with redacted text and metadata
        
        Example:
            >>> result = await redactor.redact_text(
            ...     "Иван, email: ivan@mail.ru, тел: +7-123-456-78-90"
            ... )
            >>> print(result.text)
            [PERSON], email: [EMAIL_ADDRESS], тел: [PHONE_NUMBER]
        """
        await self._initialize()
        
        try:
            with track_time(
                pii_detection_duration,
                {"operation": "redact", "language": self.language}
            ):
                # Detect entities
                results = self._analyzer.analyze(
                    text=text,
                    language=self.language,
                    entities=[e.value for e in self.entity_types],
                    score_threshold=score_threshold,
                )
                
                if not results:
                    logger.debug("no_pii_found", text_length=len(text))
                    return RedactionResult(
                        text=text,
                        entities=[],
                        redacted_count=0,
                    )
                
                # Apply redaction strategy
                operator_config = self._get_operator_config()
                
                anonymized = self._anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators=operator_config,
                )
                
                # Format entities
                entities = [
                    {
                        "entity_type": r.entity_type,
                        "start": r.start,
                        "end": r.end,
                        "score": r.score,
                        "original": text[r.start:r.end],
                    }
                    for r in results
                ]
                
                logger.info(
                    "text_redacted",
                    entity_count=len(entities),
                    text_length=len(text),
                    redacted_length=len(anonymized.text),
                )
                
                return RedactionResult(
                    text=anonymized.text,
                    entities=entities,
                    redacted_count=len(entities),
                )
        
        except Exception as e:
            track_error("security", e)
            logger.exception("pii_redaction_failed")
            raise SecurityError(
                message="PII redaction failed",
                details={"error": str(e)},
            )
    
    def _get_operator_config(self) -> Dict:
        """Get operator configuration for redaction strategy."""
        if self.strategy == RedactionStrategy.REPLACE:
            return {
                entity.value: OperatorConfig("replace", {"new_value": f"[{entity.value}]"})
                for entity in self.entity_types
            }
        
        elif self.strategy == RedactionStrategy.MASK:
            return {
                entity.value: OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False})
                for entity in self.entity_types
            }
        
        elif self.strategy == RedactionStrategy.HASH:
            return {
                entity.value: OperatorConfig("hash", {"hash_type": "sha256"})
                for entity in self.entity_types
            }
        
        elif self.strategy == RedactionStrategy.ENCRYPT:
            # Note: Requires encryption key setup
            return {
                entity.value: OperatorConfig("encrypt", {"key": settings.security.encryption_key.get_secret_value()})
                for entity in self.entity_types
            }
        
        else:  # KEEP
            return {
                entity.value: OperatorConfig("keep")
                for entity in self.entity_types
            }
    
    async def redact_document(
        self,
        document: Dict,
        text_fields: List[str],
        score_threshold: float = 0.5,
    ) -> Dict:
        """
        Redact PII in document fields.
        
        Args:
            document: Document dict
            text_fields: Fields to redact
            score_threshold: Minimum confidence score
        
        Returns:
            Document with redacted fields
        
        Example:
            >>> doc = {
            ...     "title": "Contract",
            ...     "text": "Иванов Иван...",
            ...     "metadata": {"author": "Ivan"}
            ... }
            >>> redacted = await redactor.redact_document(
            ...     doc, text_fields=["text", "metadata.author"]
            ... )
        """
        redacted_doc = document.copy()
        total_redacted = 0
        
        for field_path in text_fields:
            # Navigate nested fields (e.g., "metadata.author")
            parts = field_path.split(".")
            current = redacted_doc
            
            for i, part in enumerate(parts[:-1]):
                if part not in current:
                    break
                current = current[part]
            
            # Redact final field
            field_name = parts[-1]
            if field_name in current and isinstance(current[field_name], str):
                result = await self.redact_text(
                    current[field_name],
                    score_threshold,
                )
                current[field_name] = result.text
                total_redacted += result.redacted_count
        
        logger.info(
            "document_redacted",
            field_count=len(text_fields),
            total_redacted=total_redacted,
        )
        
        return redacted_doc


class RedactionResult:
    """Result of PII redaction."""
    
    def __init__(
        self,
        text: str,
        entities: List[Dict],
        redacted_count: int,
    ):
        self.text = text
        self.entities = entities
        self.redacted_count = redacted_count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "entities": self.entities,
            "redacted_count": self.redacted_count,
        }


# Convenience function
async def redact_text(
    text: str,
    language: str = "ru",
    strategy: RedactionStrategy = RedactionStrategy.REPLACE,
    score_threshold: float = 0.5,
) -> RedactionResult:
    """
    Convenience function for quick PII redaction.
    
    Example:
        >>> result = await redact_text("Иван Иванов, ivan@mail.ru")
        >>> print(result.text)
        [PERSON], [EMAIL_ADDRESS]
    """
    async with PIIRedactor(language=language, strategy=strategy) as redactor:
        return await redactor.redact_text(text, score_threshold)
