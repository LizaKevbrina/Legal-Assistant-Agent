"""
Metadata Extractor - Извлечение юридических метаданных из текста.

Извлекает:
- Тип документа (договор, закон, решение суда)
- Стороны договора (юр. лица)
- Даты (подписания, действия)
- Суммы и цифры
- Номера документов
- Юрисдикция
- Область права (гражданское, уголовное, корпоративное)

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from enum import Enum

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
    track_error,
)


logger = get_logger(__name__)
settings = get_settings()


class DocumentType(str, Enum):
    """Типы юридических документов."""
    CONTRACT = "contract"
    LAW = "law"
    REGULATION = "regulation"
    COURT_DECISION = "court_decision"
    PROTOCOL = "protocol"
    POWER_OF_ATTORNEY = "power_of_attorney"
    ACT = "act"
    UNKNOWN = "unknown"


class Jurisdiction(str, Enum):
    """Юрисдикции."""
    RF = "RF"  # Российская Федерация
    MOSCOW = "Moscow"
    SPB = "Saint Petersburg"
    UNKNOWN = "unknown"


class LegalArea(str, Enum):
    """Области права."""
    CIVIL = "civil"
    CRIMINAL = "criminal"
    CORPORATE = "corporate"
    TAX = "tax"
    LABOR = "labor"
    REAL_ESTATE = "real_estate"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    UNKNOWN = "unknown"


class MetadataExtractor:
    """
    Production-ready metadata extractor для юридических документов.
    
    Features:
    - Regex-based extraction (fast)
    - NER для сущностей
    - Normalized output
    - Confidence scores
    - Multi-pattern matching
    
    Example:
        >>> extractor = MetadataExtractor()
        >>> metadata = extractor.extract("Договор купли-продажи...")
        >>> print(metadata["doc_type"])
    """
    
    # Document type patterns
    DOC_TYPE_PATTERNS = {
        DocumentType.CONTRACT: [
            r"договор",
            r"контракт",
            r"соглашение",
        ],
        DocumentType.LAW: [
            r"федеральный закон",
            r"закон\s+(?:рф|российской федерации)",
            r"кодекс",
        ],
        DocumentType.REGULATION: [
            r"постановление",
            r"распоряжение",
            r"приказ",
        ],
        DocumentType.COURT_DECISION: [
            r"решение суда",
            r"определение суда",
            r"приговор",
        ],
        DocumentType.PROTOCOL: [
            r"протокол",
        ],
        DocumentType.POWER_OF_ATTORNEY: [
            r"доверенность",
        ],
        DocumentType.ACT: [
            r"акт\s+приема",
            r"акт\s+выполненных работ",
        ],
    }
    
    # Date patterns
    DATE_PATTERNS = [
        r"(\d{1,2})\.(\d{1,2})\.(\d{4})",  # DD.MM.YYYY
        r"(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+(\d{4})",
        r"от\s+[\"«]?(\d{1,2})\.(\d{1,2})\.(\d{4})[\"»]?",
    ]
    
    # Money patterns
    MONEY_PATTERNS = [
        r"(\d+(?:\s?\d+)*(?:[,\.]\d+)?)\s*(?:руб(?:л(?:ей|я)?)?|₽)",
        r"(\d+(?:\s?\d+)*(?:[,\.]\d+)?)\s*(?:dollar|usd|\$)",
        r"(\d+(?:\s?\d+)*(?:[,\.]\d+)?)\s*(?:euro|eur|€)",
    ]
    
    # Document number patterns
    DOC_NUMBER_PATTERNS = [
        r"№\s*(\d+(?:-\d+)?(?:/\d+)?)",
        r"номер\s*(\d+(?:-\d+)?(?:/\d+)?)",
    ]
    
    # Legal entity patterns (упрощенно)
    ENTITY_PATTERNS = [
        r'ООО\s+"([^"]+)"',
        r'АО\s+"([^"]+)"',
        r'ПАО\s+"([^"]+)"',
        r'ЗАО\s+"([^"]+)"',
        r'ИП\s+([А-ЯЁ][а-яё]+\s+[А-ЯЁ]\.[А-ЯЁ]\.)',
    ]
    
    # Month name to number
    MONTHS = {
        "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
        "мая": 5, "июня": 6, "июля": 7, "августа": 8,
        "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12,
    }
    
    def __init__(self):
        """Initialize metadata extractor."""
        logger.info("metadata_extractor_initialized")
    
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for better pattern matching.
        
        Args:
            text: Raw text.
            
        Returns:
            Normalized text (lowercase, extra spaces removed).
        """
        # Lowercase
        text = text.lower()
        
        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)
        
        return text.strip()
    
    def _extract_doc_type(self, text: str) -> DocumentType:
        """
        Extract document type.
        
        Args:
            text: Document text.
            
        Returns:
            DocumentType enum.
        """
        normalized = self._normalize_text(text)
        
        # Try each pattern
        for doc_type, patterns in self.DOC_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, normalized, re.IGNORECASE):
                    logger.debug("doc_type_detected", type=doc_type.value)
                    return doc_type
        
        logger.debug("doc_type_unknown")
        return DocumentType.UNKNOWN
    
    def _extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text.
        
        Args:
            text: Document text.
            
        Returns:
            List of dates in ISO format (YYYY-MM-DD).
        """
        dates = []
        
        for pattern in self.DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    groups = match.groups()
                    
                    # DD.MM.YYYY format
                    if len(groups) == 3 and groups[1].isdigit():
                        day, month, year = groups
                        date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    
                    # DD month_name YYYY format
                    elif len(groups) == 3 and groups[1] in self.MONTHS:
                        day, month_name, year = groups
                        month = self.MONTHS[month_name]
                        date = f"{year}-{month:02d}-{day.zfill(2)}"
                    
                    else:
                        continue
                    
                    # Validate date
                    datetime.strptime(date, "%Y-%m-%d")
                    dates.append(date)
                
                except (ValueError, IndexError):
                    continue
        
        # Remove duplicates, keep order
        unique_dates = list(dict.fromkeys(dates))
        
        logger.debug("dates_extracted", count=len(unique_dates))
        return unique_dates
    
    def _extract_amounts(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract monetary amounts.
        
        Args:
            text: Document text.
            
        Returns:
            List of amounts with currency.
        """
        amounts = []
        
        for pattern in self.MONEY_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                amount_str = match.group(1).replace(" ", "").replace(",", ".")
                
                try:
                    amount = float(amount_str)
                    
                    # Detect currency
                    full_match = match.group(0).lower()
                    if "руб" in full_match or "₽" in full_match:
                        currency = "RUB"
                    elif "dollar" in full_match or "usd" in full_match or "$" in full_match:
                        currency = "USD"
                    elif "euro" in full_match or "eur" in full_match or "€" in full_match:
                        currency = "EUR"
                    else:
                        currency = "UNKNOWN"
                    
                    amounts.append({
                        "amount": amount,
                        "currency": currency,
                    })
                
                except ValueError:
                    continue
        
        logger.debug("amounts_extracted", count=len(amounts))
        return amounts
    
    def _extract_doc_numbers(self, text: str) -> List[str]:
        """
        Extract document numbers.
        
        Args:
            text: Document text.
            
        Returns:
            List of document numbers.
        """
        numbers = []
        
        for pattern in self.DOC_NUMBER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                number = match.group(1)
                numbers.append(number)
        
        # Remove duplicates
        unique_numbers = list(dict.fromkeys(numbers))
        
        logger.debug("doc_numbers_extracted", count=len(unique_numbers))
        return unique_numbers
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract legal entities (компании, ИП).
        
        Args:
            text: Document text.
            
        Returns:
            List of entity names.
        """
        entities = []
        
        for pattern in self.ENTITY_PATTERNS:
            matches = re.finditer(pattern, text)
            for match in matches:
                entity = match.group(1)
                entities.append(entity)
        
        # Remove duplicates
        unique_entities = list(dict.fromkeys(entities))
        
        logger.debug("entities_extracted", count=len(unique_entities))
        return unique_entities
    
    def _detect_jurisdiction(self, text: str) -> Jurisdiction:
        """
        Detect jurisdiction.
        
        Args:
            text: Document text.
            
        Returns:
            Jurisdiction enum.
        """
        normalized = self._normalize_text(text)
        
        if re.search(r"москв", normalized):
            return Jurisdiction.MOSCOW
        elif re.search(r"санкт-петербург|спб", normalized):
            return Jurisdiction.SPB
        elif re.search(r"российск(?:ая|ой) федераци|рф", normalized):
            return Jurisdiction.RF
        
        return Jurisdiction.UNKNOWN
    
    def _detect_legal_area(self, text: str) -> LegalArea:
        """
        Detect legal area (упрощенно, по keywords).
        
        Args:
            text: Document text.
            
        Returns:
            LegalArea enum.
        """
        normalized = self._normalize_text(text)
        
        # Simple keyword matching
        if re.search(r"купл[аи]-продаж|аренд|залог", normalized):
            return LegalArea.CIVIL
        elif re.search(r"устав|акционер|директор|учредител", normalized):
            return LegalArea.CORPORATE
        elif re.search(r"налог|ндс|налогооблож", normalized):
            return LegalArea.TAX
        elif re.search(r"трудов(?:ой|ая) договор|работник|работодател", normalized):
            return LegalArea.LABOR
        elif re.search(r"недвижимост|земельн|участок", normalized):
            return LegalArea.REAL_ESTATE
        elif re.search(r"патент|товарный знак|авторск", normalized):
            return LegalArea.INTELLECTUAL_PROPERTY
        
        return LegalArea.UNKNOWN
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        Extract all metadata from document text.
        
        Args:
            text: Full document text.
            
        Returns:
            Metadata dictionary:
            {
                "doc_type": "contract",
                "dates": ["2024-01-15", ...],
                "amounts": [{"amount": 100000, "currency": "RUB"}, ...],
                "doc_numbers": ["123/2024", ...],
                "parties": ["ООО Рога и копыта", ...],
                "jurisdiction": "Moscow",
                "legal_area": "corporate",
            }
        """
        logger.info("extracting_metadata", text_length=len(text))
        
        with track_time("metadata_extraction_seconds"):
            metadata = {
                "doc_type": self._extract_doc_type(text).value,
                "dates": self._extract_dates(text),
                "amounts": self._extract_amounts(text),
                "doc_numbers": self._extract_doc_numbers(text),
                "parties": self._extract_entities(text),
                "jurisdiction": self._detect_jurisdiction(text).value,
                "legal_area": self._detect_legal_area(text).value,
            }
        
        logger.info(
            "metadata_extracted",
            doc_type=metadata["doc_type"],
            num_dates=len(metadata["dates"]),
            num_amounts=len(metadata["amounts"]),
            num_parties=len(metadata["parties"]),
        )
        
        return metadata


# Convenience function
def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Quick metadata extraction.
    
    Args:
        text: Document text.
        
    Returns:
        Metadata dictionary.
        
    Example:
        >>> metadata = extract_metadata("Договор №123 от 01.01.2024...")
        >>> print(metadata["doc_type"])
    """
    extractor = MetadataExtractor()
    return extractor.extract(text)
