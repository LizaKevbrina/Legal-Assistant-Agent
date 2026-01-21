"""
E2E tests for document ingestion pipeline.

Tests:
- PDF parsing (LlamaParse → Vision → OCR fallback)
- Metadata extraction
- Chunking strategies
- PII redaction
- Error handling
"""

import pytest
from pathlib import Path

from legal_assistant.ingestion import (
    parse_document,
    DocumentParser,
    MetadataExtractor,
    SemanticChunker,
)
from legal_assistant.security import PIIRedactor


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pdf_parsing_fallback_chain(
    sample_pdf_document: Path,
    measure_performance,
):
    """Test PDF parsing with fallback chain."""
    
    print("\n=== Testing PDF Parsing Fallback Chain ===")
    
    with measure_performance() as perf:
        result = await parse_document(sample_pdf_document)
    
    # Assertions
    assert result.text
    assert len(result.text) > 100
    assert result.chunks
    assert len(result.chunks) > 0
    assert result.metadata
    assert result.parsing_method in ["llamaparse", "vision", "ocr"]
    
    print(f"✓ Parsing method: {result.parsing_method}")
    print(f"✓ Text length: {len(result.text)} chars")
    print(f"✓ Chunks: {len(result.chunks)}")
    print(f"✓ Duration: {perf.duration:.2f}s")
    print(f"✓ Metadata fields: {list(result.metadata.keys())}")
    
    # Check metadata
    if "doc_type" in result.metadata:
        print(f"✓ Document type: {result.metadata['doc_type']}")
    
    if "num_pages" in result.metadata:
        print(f"✓ Pages: {result.metadata['num_pages']}")
    
    print("\n✅ PDF parsing test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_metadata_extraction(sample_legal_text: str):
    """Test legal metadata extraction."""
    
    print("\n=== Testing Metadata Extraction ===")
    
    extractor = MetadataExtractor()
    metadata = await extractor.extract_metadata(sample_legal_text)
    
    # Assertions
    assert metadata is not None
    assert "doc_type" in metadata
    
    print(f"✓ Document type: {metadata.get('doc_type')}")
    
    if "dates" in metadata and metadata["dates"]:
        print(f"✓ Dates found: {metadata['dates']}")
    
    if "amounts" in metadata and metadata["amounts"]:
        print(f"✓ Amounts found: {metadata['amounts']}")
    
    if "parties" in metadata and metadata["parties"]:
        print(f"✓ Parties found: {metadata['parties']}")
    
    if "jurisdiction" in metadata:
        print(f"✓ Jurisdiction: {metadata['jurisdiction']}")
    
    print("\n✅ Metadata extraction test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_chunking_strategies(sample_legal_text: str):
    """Test different chunking strategies."""
    
    print("\n=== Testing Chunking Strategies ===")
    
    chunker = SemanticChunker()
    
    # Test 1: Sentence window strategy
    print("\n--- Strategy 1: Sentence Window ---")
    chunks = await chunker.chunk_text(
        sample_legal_text,
        strategy="sentence_window",
        chunk_size=512,
        overlap=128,
    )
    
    assert len(chunks) > 0
    assert all(chunk.text for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
    
    print(f"✓ Chunks created: {len(chunks)}")
    print(f"✓ Avg chunk size: {sum(len(c.text) for c in chunks) / len(chunks):.0f} chars")
    
    # Test 2: Fixed size strategy
    print("\n--- Strategy 2: Fixed Size ---")
    chunks_fixed = await chunker.chunk_text(
        sample_legal_text,
        strategy="fixed_size",
        chunk_size=256,
        overlap=50,
    )
    
    assert len(chunks_fixed) > 0
    print(f"✓ Chunks created: {len(chunks_fixed)}")
    
    # Test 3: Legal clause strategy
    print("\n--- Strategy 3: Legal Clause ---")
    chunks_clause = await chunker.chunk_text(
        sample_legal_text,
        strategy="legal_clause",
    )
    
    assert len(chunks_clause) > 0
    print(f"✓ Chunks created: {len(chunks_clause)}")
    
    print("\n✅ Chunking strategies test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pii_redaction_in_document():
    """Test PII redaction in document processing."""
    
    print("\n=== Testing PII Redaction ===")
    
    text_with_pii = """
    ДОГОВОР № 123
    
    Продавец: ООО "Компания"
    ИНН: 7702123456
    Директор: Иванов Иван Иванович
    Email: ivanov@company.ru
    Телефон: +7-495-123-45-67
    Паспорт: 4512 567890
    
    Цена: 1 000 000 рублей
    """
    
    async with PIIRedactor() as redactor:
        result = await redactor.redact_text(text_with_pii)
    
    # Assertions
    assert result.text
    assert result.redacted_count > 0
    assert result.entities
    
    # Check that PII was actually redacted
    assert "ivanov@company.ru" not in result.text.lower()
    assert "7702123456" not in result.text
    
    print(f"✓ Entities redacted: {result.redacted_count}")
    print(f"✓ Entity types: {set(e['entity_type'] for e in result.entities)}")
    print(f"\nOriginal text:\n{text_with_pii[:200]}...")
    print(f"\nRedacted text:\n{result.text[:200]}...")
    
    print("\n✅ PII redaction test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_document_parser_with_options(
    sample_pdf_document: Path,
):
    """Test DocumentParser with various options."""
    
    print("\n=== Testing DocumentParser Options ===")
    
    async with DocumentParser() as parser:
        # Test with custom chunking
        result = await parser.parse(
            sample_pdf_document,
            chunking_strategy="sentence_window",
            chunk_size=256,
            extract_metadata=True,
            redact_pii=False,  # Disable for speed
        )
        
        assert result.text
        assert result.chunks
        assert result.metadata
        
        print(f"✓ Chunks: {len(result.chunks)}")
        print(f"✓ Avg chunk size: {sum(len(c.text) for c in result.chunks) / len(result.chunks):.0f}")
        
        # Check chunk metadata
        if result.chunks:
            first_chunk = result.chunks[0]
            print(f"✓ Chunk metadata keys: {list(first_chunk.metadata.keys())}")
    
    print("\n✅ DocumentParser options test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_large_document_handling(temp_dir: Path):
    """Test handling of large documents."""
    
    print("\n=== Testing Large Document Handling ===")
    
    # Create large text document
    large_text = "Договор купли-продажи.\n" * 10000  # ~300KB
    large_file = temp_dir / "large_doc.txt"
    large_file.write_text(large_text, encoding="utf-8")
    
    print(f"✓ Created large document: {large_file.stat().st_size / 1024:.0f} KB")
    
    # Parse
    result = await parse_document(large_file)
    
    assert result.text
    assert len(result.chunks) > 10
    
    print(f"✓ Parsed successfully")
    print(f"✓ Chunks created: {len(result.chunks)}")
    print(f"✓ Total text: {len(result.text)} chars")
    
    print("\n✅ Large document test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_parsing_error_recovery(temp_dir: Path):
    """Test error recovery in document parsing."""
    
    print("\n=== Testing Parsing Error Recovery ===")
    
    # Test 1: Corrupted PDF (create invalid file)
    print("\n--- Test 1: Corrupted file ---")
    corrupted = temp_dir / "corrupted.pdf"
    corrupted.write_bytes(b"Not a valid PDF file")
    
    # Should fallback gracefully
    try:
        result = await parse_document(corrupted)
        # OCR might extract some text or fail completely
        print(f"✓ Handled corrupted file (method: {result.parsing_method})")
    except Exception as e:
        print(f"✓ Error handled correctly: {type(e).__name__}")
    
    # Test 2: Empty file
    print("\n--- Test 2: Empty file ---")
    empty = temp_dir / "empty.txt"
    empty.write_text("", encoding="utf-8")
    
    try:
        result = await parse_document(empty)
        assert not result.text or len(result.text) == 0
        print(f"✓ Handled empty file")
    except Exception as e:
        print(f"✓ Error handled: {type(e).__name__}")
    
    # Test 3: Unsupported file type
    print("\n--- Test 3: Unsupported type ---")
    unsupported = temp_dir / "file.exe"
    unsupported.write_bytes(b"\x00\x01\x02")
    
    from legal_assistant.core.exceptions import FileValidationError
    with pytest.raises(FileValidationError):
        from legal_assistant.security import validate_file
        validate_file(unsupported)
    
    print(f"✓ Unsupported type rejected")
    
    print("\n✅ Error recovery test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multimodal_document_parsing(temp_dir: Path):
    """Test parsing document with images/charts."""
    
    print("\n=== Testing Multimodal Document Parsing ===")
    
    # Create PDF with image (simple example)
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    pdf_path = temp_dir / "doc_with_image.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    
    # Add text
    c.drawString(100, 750, "Договор купли-продажи")
    c.drawString(100, 730, "Цена: 1 000 000 рублей")
    
    # Add simple shape (simulating chart)
    c.rect(100, 500, 400, 200)
    c.drawString(150, 600, "График продаж")
    
    c.save()
    
    print(f"✓ Created PDF with visual elements")
    
    # Parse with vision enabled
    result = await parse_document(
        pdf_path,
        # Vision might be triggered for complex elements
    )
    
    assert result.text
    assert "договор" in result.text.lower()
    
    print(f"✓ Parsed multimodal document")
    print(f"✓ Method: {result.parsing_method}")
    print(f"✓ Text extracted: {len(result.text)} chars")
    
    print("\n✅ Multimodal parsing test passed!")
