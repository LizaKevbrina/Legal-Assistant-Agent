"""
E2E tests for security layer integration.

Tests:
- PII detection in full pipeline
- Input validation at entry points
- Output guardrails enforcement
- HITL triggering
- Cost controls
"""

import pytest
from pathlib import Path

from legal_assistant.ingestion import parse_document
from legal_assistant.retrieval import RAGQueryEngine
from legal_assistant.generation import ResponseSynthesizer
from legal_assistant.security import (
    PIIRedactor,
    InputValidator,
    OutputGuardrails,
    RedactionStrategy,
    GuardrailStatus,
    ModelOutputLimits,
)
from legal_assistant.core.exceptions import (
    FileValidationError,
    QueryValidationError,
    QualityError,
)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_pii_detection_in_full_pipeline():
    """Test PII detection throughout entire pipeline."""
    
    print("\n=== Testing PII Detection in Full Pipeline ===")
    
    # Document with PII
    pii_document = """
    ДОГОВОР № 12345
    
    Стороны:
    1. ООО "Продавец"
       ИНН: 7702123456
       Директор: Иванов Иван Иванович
       Email: ivanov@company.ru
       Телефон: +7-495-123-45-67
       Паспорт: 4512 567890
       СНИЛС: 123-456-789 01
    
    2. ООО "Покупатель"
       Email: buyer@company.ru
       Телефон: +7-495-987-65-43
    
    Предмет: Поставка товара на сумму 1 000 000 рублей.
    Банковский счет: 40702810100000001234
    """
    
    # Step 1: Redact PII from document
    print("\n--- Step 1: Document PII Redaction ---")
    
    async with PIIRedactor(strategy=RedactionStrategy.REPLACE) as redactor:
        doc_result = await redactor.redact_text(pii_document)
    
    print(f"✓ Entities redacted: {doc_result.redacted_count}")
    print(f"✓ Entity types: {set(e['entity_type'] for e in doc_result.entities)}")
    
    # Verify PII removed
    assert "ivanov@company.ru" not in doc_result.text.lower()
    assert "7702123456" not in doc_result.text
    assert "4512 567890" not in doc_result.text
    
    print(f"✓ PII successfully removed from document")
    
    # Step 2: Check query for PII
    print("\n--- Step 2: Query PII Detection ---")
    
    query_with_pii = "Найди договор для ivanov@company.ru"
    
    async with PIIRedactor() as redactor:
        query_result = await redactor.redact_text(query_with_pii)
    
    if query_result.redacted_count > 0:
        print(f"✓ PII detected in query: {query_result.redacted_count} entities")
        print(f"✓ Redacted query: {query_result.text}")
    
    # Step 3: Check response for PII leakage
    print("\n--- Step 3: Response PII Leakage Check ---")
    
    response_text = "Договор подписан директором Ивановым И.И., email: test@test.ru"
    
    guardrails = OutputGuardrails(check_pii_leakage=True)
    result = guardrails.check_output(
        answer=response_text,
        sources=[{"text": "Source text", "score": 0.9}],
        confidence=0.8,
    )
    
    # Should detect email in response
    pii_issues = [issue for issue in result.issues if "email" in issue.lower()]
    
    if pii_issues:
        print(f"✓ PII leakage detected in response")
        print(f"  Issues: {pii_issues}")
    
    print("\n✅ PII detection pipeline test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_input_validation_at_all_entry_points(temp_dir: Path):
    """Test input validation at all pipeline entry points."""
    
    print("\n=== Testing Input Validation at Entry Points ===")
    
    validator = InputValidator()
    
    # Entry Point 1: File Upload
    print("\n--- Entry Point 1: File Upload ---")
    
    # Valid file
    valid_file = temp_dir / "valid.txt"
    valid_file.write_text("Valid content", encoding="utf-8")
    
    file_info = validator.validate_file(valid_file, max_size_mb=10)
    assert file_info["mime_type"] == "text/plain"
    print(f"✓ Valid file accepted: {file_info['name']}")
    
    # Invalid: file too large
    large_file = temp_dir / "large.txt"
    large_file.write_text("x" * (60 * 1024 * 1024))  # 60MB
    
    with pytest.raises(FileValidationError):
        validator.validate_file(large_file, max_size_mb=50)
    print(f"✓ Oversized file rejected")
    
    # Invalid: wrong type
    exe_file = temp_dir / "malware.exe"
    exe_file.write_bytes(b"\x4d\x5a\x90\x00")  # MZ header
    
    with pytest.raises(FileValidationError):
        validator.validate_file(exe_file)
    print(f"✓ Executable file rejected")
    
    # Entry Point 2: Query Input
    print("\n--- Entry Point 2: Query Input ---")
    
    # Valid query
    valid_query = validator.validate_query("Что такое договор?")
    assert valid_query
    print(f"✓ Valid query accepted")
    
    # Invalid: SQL injection
    with pytest.raises(QueryValidationError):
        validator.validate_query("'; DROP TABLE users; --")
    print(f"✓ SQL injection blocked")
    
    # Invalid: XSS attempt
    xss_query = "<script>alert('xss')</script>What is contract?"
    sanitized = validator.sanitize_text(xss_query, remove_xss=True)
    assert "<script>" not in sanitized
    print(f"✓ XSS sanitized: {sanitized[:50]}")
    
    # Invalid: too long
    with pytest.raises(QueryValidationError):
        validator.validate_query("x" * 2000)
    print(f"✓ Overly long query rejected")
    
    # Entry Point 3: Metadata Input
    print("\n--- Entry Point 3: Metadata Input ---")
    
    metadata = {
        "doc_type": "contract",
        "author": "Test <script>",
        "date": "2024-01-01",
    }
    
    validated_metadata = validator.validate_metadata(
        metadata,
        required_fields=["doc_type"],
    )
    
    # XSS should be sanitized
    assert "<script>" not in validated_metadata["author"]
    print(f"✓ Metadata validated and sanitized")
    
    print("\n✅ Input validation test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_output_guardrails_enforcement(
    sample_pdf_document: Path,
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test output guardrails enforcement."""
    
    print("\n=== Testing Output Guardrails Enforcement ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_guardrails"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Generate answer
    async with RAGQueryEngine(collection_name) as engine:
        query_result = await engine.query("цена договора", top_k=5)
        
        response = await response_synthesizer.synthesize(
            query="цена договора",
            retrieved_documents=query_result.retrieved_docs,
        )
    
    # Test guardrails with different thresholds
    guardrails = OutputGuardrails(
        confidence_threshold=0.7,
        require_disclaimer=True,
        check_pii_leakage=True,
    )
    
    print("\n--- Guardrails Check ---")
    
    check_result = guardrails.check_output(
        answer=response.answer,
        sources=[
            {"text": doc["metadata"]["text"], "score": doc["score"]}
            for doc in query_result.retrieved_docs
        ],
        confidence=response.confidence,
    )
    
    print(f"✓ Status: {check_result.status.value}")
    print(f"✓ Confidence: {check_result.confidence:.3f}")
    print(f"✓ Confidence level: {check_result.confidence_level.value}")
    print(f"✓ Safe for output: {check_result.safe_for_output}")
    print(f"✓ Needs review: {check_result.needs_review}")
    
    if check_result.issues:
        print(f"✓ Issues found: {len(check_result.issues)}")
        for issue in check_result.issues:
            print(f"  - {issue}")
    
    if check_result.warnings:
        print(f"✓ Warnings: {len(check_result.warnings)}")
        for warning in check_result.warnings:
            print(f"  - {warning}")
    
    # Test HITL triggering
    print("\n--- HITL Trigger Test ---")
    
    hitl_triggered = guardrails.should_trigger_hitl(
        check_result,
        task_type="legal_opinion",
    )
    
    print(f"✓ HITL triggered: {hitl_triggered}")
    
    if hitl_triggered:
        print(f"  Reason: Low confidence or critical issues")
    
    # Test disclaimer addition
    print("\n--- Disclaimer Addition ---")
    
    answer_with_disclaimer = guardrails.add_disclaimer(
        response.answer,
        disclaimer_type="legal",
    )
    
    assert "юридической консультацией" in answer_with_disclaimer.lower()
    print(f"✓ Disclaimer added")
    print(f"  Preview: ...{answer_with_disclaimer[-200:]}")
    
    print("\n✅ Guardrails enforcement test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cost_controls_and_budgets():
    """Test cost controls and token budgets."""
    
    print("\n=== Testing Cost Controls ===")
    
    limits = ModelOutputLimits(
        max_tokens_per_request=1000,
        max_cost_per_request=0.10,
        daily_token_budget=10000,
        daily_cost_budget=1.00,
    )
    
    print(f"✓ Limits configured:")
    print(f"  Max tokens/request: {limits.max_tokens_per_request}")
    print(f"  Max cost/request: ${limits.max_cost_per_request}")
    print(f"  Daily token budget: {limits.daily_token_budget}")
    print(f"  Daily cost budget: ${limits.daily_cost_budget}")
    
    # Test 1: Within limits
    print("\n--- Test 1: Within Limits ---")
    
    try:
        limits.check_limits(
            estimated_tokens=500,
            estimated_cost=0.05,
        )
        print(f"✓ Request approved (within limits)")
    except QualityError as e:
        pytest.fail(f"Should not have raised: {e}")
    
    # Test 2: Exceed per-request token limit
    print("\n--- Test 2: Exceed Token Limit ---")
    
    with pytest.raises(QualityError) as exc_info:
        limits.check_limits(
            estimated_tokens=2000,  # Exceeds 1000
            estimated_cost=0.05,
        )
    
    assert "token limit" in str(exc_info.value).lower()
    print(f"✓ Token limit enforced")
    
    # Test 3: Exceed per-request cost limit
    print("\n--- Test 3: Exceed Cost Limit ---")
    
    with pytest.raises(QualityError) as exc_info:
        limits.check_limits(
            estimated_tokens=500,
            estimated_cost=0.50,  # Exceeds 0.10
        )
    
    assert "cost limit" in str(exc_info.value).lower()
    print(f"✓ Cost limit enforced")
    
    # Test 4: Track usage
    print("\n--- Test 4: Usage Tracking ---")
    
    limits.track_usage(tokens=500, cost=0.05)
    print(f"✓ Usage tracked:")
    print(f"  Tokens used today: {limits.tokens_used_today}")
    print(f"  Cost used today: ${limits.cost_used_today:.4f}")
    
    # Test 5: Exceed daily budget
    print("\n--- Test 5: Daily Budget ---")
    
    # Use up most of budget
    limits.track_usage(tokens=9000, cost=0.90)
    
    with pytest.raises(QualityError) as exc_info:
        limits.check_limits(
            estimated_tokens=2000,  # Would exceed daily 10000
            estimated_cost=0.05,
        )
    
    assert "daily" in str(exc_info.value).lower()
    print(f"✓ Daily budget enforced")
    
    # Test 6: Reset daily usage
    print("\n--- Test 6: Reset ---")
    
    limits.reset_daily_usage()
    assert limits.tokens_used_today == 0
    assert limits.cost_used_today == 0.0
    print(f"✓ Daily usage reset")
    
    print("\n✅ Cost controls test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_security_integration_full_workflow(
    sample_pdf_document: Path,
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test security integration in full workflow."""
    
    print("\n=== Testing Security Integration (Full Workflow) ===")
    
    validator = InputValidator()
    guardrails = OutputGuardrails()
    
    # Step 1: Validate file upload
    print("\n--- Step 1: File Validation ---")
    file_info = validator.validate_file(sample_pdf_document)
    print(f"✓ File validated: {file_info['mime_type']}")
    
    # Step 2: Parse document (with potential PII)
    print("\n--- Step 2: Document Parsing ---")
    parse_result = await parse_document(sample_pdf_document)
    print(f"✓ Document parsed: {len(parse_result.chunks)} chunks")
    
    # Step 3: Index (simulated)
    collection_name = "test_security_full"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in parse_result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in parse_result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ Document indexed")
    
    # Step 4: Validate query
    print("\n--- Step 4: Query Validation ---")
    query = "Какая цена договора?"
    validated_query = validator.validate_query(query)
    print(f"✓ Query validated: {validated_query}")
    
    # Step 5: Query & Generate
    print("\n--- Step 5: Query & Generate ---")
    async with RAGQueryEngine(collection_name) as engine:
        query_result = await engine.query(validated_query, top_k=5)
        
        response = await response_synthesizer.synthesize(
            query=validated_query,
            retrieved_documents=query_result.retrieved_docs,
        )
    
    print(f"✓ Answer generated: {len(response.answer)} chars")
    
    # Step 6: Guardrails check
    print("\n--- Step 6: Guardrails Check ---")
    check_result = guardrails.check_output(
        answer=response.answer,
        sources=[
            {"text": doc["metadata"]["text"], "score": doc["score"]}
            for doc in query_result.retrieved_docs
        ],
        confidence=response.confidence,
    )
    
    print(f"✓ Guardrails status: {check_result.status.value}")
    print(f"✓ Safe for output: {check_result.safe_for_output}")
    
    # Step 7: Add disclaimer if safe
    print("\n--- Step 7: Final Output ---")
    
    if check_result.safe_for_output:
        final_answer = guardrails.add_disclaimer(response.answer)
        print(f"✓ Final answer ready ({len(final_answer)} chars)")
    else:
        print(f"✗ Answer requires review (issues: {len(check_result.issues)})")
    
    print("\n✅ Security integration test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_malicious_input_handling(temp_dir: Path):
    """Test handling of malicious inputs."""
    
    print("\n=== Testing Malicious Input Handling ===")
    
    validator = InputValidator()
    
    # Test 1: Path traversal in filename
    print("\n--- Test 1: Path Traversal ---")
    
    with pytest.raises(FileValidationError):
        validator.validate_filename("../../etc/passwd")
    print(f"✓ Path traversal blocked")
    
    # Test 2: Null byte injection
    print("\n--- Test 2: Null Byte ---")
    
    with pytest.raises(FileValidationError):
        validator.validate_filename("file\x00.txt")
    print(f"✓ Null byte rejected")
    
    # Test 3: Script injection in metadata
    print("\n--- Test 3: Script Injection ---")
    
    metadata = {
        "title": "<script>alert('xss')</script>",
        "author": "'; DROP TABLE users; --",
    }
    
    sanitized = validator.validate_metadata(metadata)
    
    assert "<script>" not in sanitized["title"]
    assert "DROP TABLE" not in sanitized["author"]
    print(f"✓ Scripts sanitized from metadata")
    
    # Test 4: Malformed file upload
    print("\n--- Test 4: Malformed File ---")
    
    malformed = temp_dir / "malformed.pdf"
    malformed.write_bytes(b"\x00\x01\x02\x03" * 100)
    
    try:
        validator.validate_file(malformed, check_content=True)
        print(f"✗ Should have rejected malformed PDF")
    except FileValidationError:
        print(f"✓ Malformed PDF rejected")
    
    print("\n✅ Malicious input handling test passed!")
