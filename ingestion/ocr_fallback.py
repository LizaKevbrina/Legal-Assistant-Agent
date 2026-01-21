"""
OCR Fallback - Tesseract для резервного распознавания текста.

Используется как последний уровень fallback когда:
- LlamaParse недоступен
- GPT-4V не справился
- Простые сканы без сложного форматирования

Автор: AI Legal Assistant Team
Дата: 2025-01-16
"""

import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from legal_assistant.core import (
    get_logger,
    get_settings,
    DocumentParsingError,
    track_time,
    track_error,
)


logger = get_logger(__name__)
settings = get_settings()


class OCRError(DocumentParsingError):
    """OCR processing error."""
    pass


class OCRFallback:
    """
    Production-ready Tesseract OCR fallback.
    
    Features:
    - PDF to image conversion
    - Multi-language support
    - Image preprocessing
    - Page-by-page processing
    - Error handling
    
    Example:
        >>> ocr = OCRFallback()
        >>> result = ocr.process_document("scan.pdf")
        >>> print(result["text"])
    """
    
    # Tesseract configuration
    DEFAULT_LANG = "rus+eng"  # Russian + English
    DEFAULT_CONFIG = "--psm 3 --oem 3"  # Auto page segmentation, LSTM OCR
    
    # Image processing
    DPI = 300  # DPI for PDF conversion
    MAX_IMAGE_SIZE = 4096  # Max dimension in pixels
    
    def __init__(
        self,
        lang: Optional[str] = None,
        tesseract_cmd: Optional[str] = None,
    ):
        """
        Initialize OCR fallback.
        
        Args:
            lang: Tesseract language codes (e.g., "rus+eng").
            tesseract_cmd: Path to tesseract executable.
        """
        self.lang = lang or self.DEFAULT_LANG
        
        # Set tesseract path if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Verify tesseract is installed
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(
                "ocr_fallback_initialized",
                version=str(version),
                lang=self.lang,
            )
        except Exception as e:
            logger.error("tesseract_not_found", error=str(e))
            raise OCRError(
                "Tesseract not installed or not found in PATH",
                details={"error": str(e)},
            ) from e
    
    def _validate_file(self, file_path: Path) -> None:
        """
        Validate input file.
        
        Args:
            file_path: Path to file.
            
        Raises:
            DocumentParsingError: If invalid.
        """
        if not file_path.exists():
            raise DocumentParsingError(
                f"File not found: {file_path}",
                details={"file_path": str(file_path)},
            )
        
        # Check extension
        valid_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        if file_path.suffix.lower() not in valid_extensions:
            raise DocumentParsingError(
                f"Unsupported file type: {file_path.suffix}",
                details={
                    "file_path": str(file_path),
                    "extension": file_path.suffix,
                    "valid_extensions": list(valid_extensions),
                },
            )
        
        logger.debug("file_validated", file_path=str(file_path))
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR quality.
        
        Args:
            image: PIL Image.
            
        Returns:
            Preprocessed image.
        """
        # Convert to grayscale
        if image.mode != "L":
            image = image.convert("L")
        
        # Resize if too large
        max_dim = self.MAX_IMAGE_SIZE
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.debug("image_resized", new_size=new_size)
        
        return image
    
    def _ocr_image(self, image: Image.Image) -> str:
        """
        Run OCR on single image.
        
        Args:
            image: PIL Image.
            
        Returns:
            Extracted text.
            
        Raises:
            OCRError: If OCR fails.
        """
        try:
            # Preprocess
            image = self._preprocess_image(image)
            
            # Run OCR
            with track_time("ocr_processing_seconds"):
                text = pytesseract.image_to_string(
                    image,
                    lang=self.lang,
                    config=self.DEFAULT_CONFIG,
                )
            
            return text.strip()
        
        except Exception as e:
            track_error("ocr_processing")
            raise OCRError(
                f"OCR processing failed: {e}",
                details={"error": str(e)},
            ) from e
    
    def _pdf_to_images(self, pdf_path: Path) -> List[Image.Image]:
        """
        Convert PDF to images.
        
        Args:
            pdf_path: Path to PDF.
            
        Returns:
            List of PIL Images (one per page).
            
        Raises:
            OCRError: If conversion fails.
        """
        try:
            logger.info("converting_pdf_to_images", pdf_path=str(pdf_path))
            
            with track_time("pdf_conversion_seconds"):
                images = convert_from_path(
                    str(pdf_path),
                    dpi=self.DPI,
                    fmt="jpeg",
                )
            
            logger.info(
                "pdf_converted",
                pdf_path=str(pdf_path),
                num_pages=len(images),
            )
            
            return images
        
        except Exception as e:
            track_error("pdf_conversion")
            raise OCRError(
                f"PDF conversion failed: {e}",
                details={
                    "pdf_path": str(pdf_path),
                    "error": str(e),
                },
            ) from e
    
    def process_image(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Process single image file.
        
        Args:
            file_path: Path to image.
            
        Returns:
            OCR result:
            {
                "text": "Extracted text",
                "metadata": {
                    "method": "tesseract",
                    "lang": "rus+eng",
                    ...
                },
            }
            
        Raises:
            OCRError: If processing fails.
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        logger.info("starting_ocr", file_path=str(file_path))
        
        try:
            # Load image
            image = Image.open(file_path)
            
            # OCR
            text = self._ocr_image(image)
            
            logger.info(
                "ocr_completed",
                file_path=str(file_path),
                text_length=len(text),
            )
            
            return {
                "text": text,
                "metadata": {
                    "method": "tesseract",
                    "lang": self.lang,
                    "file_path": str(file_path),
                    "image_size": image.size,
                },
            }
        
        except Exception as e:
            track_error("ocr_image_processing")
            raise OCRError(
                f"Image OCR failed: {e}",
                details={
                    "file_path": str(file_path),
                    "error": str(e),
                },
            ) from e
    
    def process_pdf(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Process PDF file (converts to images first).
        
        Args:
            file_path: Path to PDF.
            
        Returns:
            OCR result with combined text from all pages.
            
        Raises:
            OCRError: If processing fails.
        """
        file_path = Path(file_path)
        self._validate_file(file_path)
        
        logger.info("starting_pdf_ocr", file_path=str(file_path))
        
        # Convert PDF to images
        images = self._pdf_to_images(file_path)
        
        # OCR each page
        page_texts = []
        for i, image in enumerate(images, 1):
            try:
                text = self._ocr_image(image)
                page_texts.append(text)
                logger.debug(
                    "page_processed",
                    page=i,
                    text_length=len(text),
                )
            except Exception as e:
                logger.warning(
                    "page_ocr_failed",
                    page=i,
                    error=str(e),
                )
                page_texts.append("")
        
        # Combine
        full_text = "\n\n--- PAGE BREAK ---\n\n".join(page_texts)
        
        logger.info(
            "pdf_ocr_completed",
            file_path=str(file_path),
            num_pages=len(images),
            total_length=len(full_text),
        )
        
        return {
            "text": full_text,
            "metadata": {
                "method": "tesseract",
                "lang": self.lang,
                "file_path": str(file_path),
                "num_pages": len(images),
            },
        }
    
    def process_document(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Process any supported document (auto-detects type).
        
        Args:
            file_path: Path to document.
            
        Returns:
            OCR result.
            
        Raises:
            OCRError: If processing fails.
        """
        file_path = Path(file_path)
        
        # Route by extension
        if file_path.suffix.lower() == ".pdf":
            return self.process_pdf(file_path)
        else:
            return self.process_image(file_path)


# Convenience functions
def process_document(file_path: str | Path, **kwargs) -> Dict[str, Any]:
    """
    Quick OCR processing.
    
    Args:
        file_path: Path to document.
        **kwargs: Passed to OCRFallback.
        
    Returns:
        OCR result.
        
    Example:
        >>> result = process_document("scan.pdf")
        >>> print(result["text"])
    """
    ocr = OCRFallback(**kwargs)
    return ocr.process_document(file_path)
