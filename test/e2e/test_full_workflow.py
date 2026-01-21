"""
Complete E2E workflow test.

Tests the full pipeline:
1. Document upload → Parse → Chunk → Embed → Index
2. Query → Retrieve → Generate → Guardrails → Response
3. Evaluation → Metrics tracking
"""

import pytest
from pathlib import Path

from legal_assistant.ingestion import parse_document
from legal_assistant.retrieval import RAGQueryEngine
from legal_assistant.generation import ResponseSynthesizer
from legal_assistant.security import (
    validate_file,
    validate_query,
    check_output,
)
from legal_assistant.evaluation import (
    EvaluationSample,
    evaluate_rag,
    record_metric,
)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.slow
async def test_complete_rag_workflow(
    sample_pdf_document: Path,
    sample_queries: list[str],
    sample_ground_truths: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
    output_guardrails,
    ragas_evaluator,
    assert_metric_in_range,
    measure_performance,
):
    """
    Test complete RAG workflow from document to evaluated answer.
    
    Flow:
    1. Validate & parse PDF
    2. Chunk & embed
    3. Index in Qdrant
    4. Query & retrieve
    5. Generate answer
    6. Guardrails check
    7. RAGAS evaluation
    8. Record metrics
    """
    
    # ==================== STEP 1: Document Ingestion ====================
    
    print("\n=== STEP 1: Document Ingestion ===")
    
    # Validate file
    file_info = validate_file(sample_pdf_document)
    assert file_info["mime_type"] == "application/pdf"
    print(f"✓ File validated: {file_info['size_mb']:.2f} MB")
    
    # Parse document
    with measure_performance() as perf:
        parse_result = await parse_document(sample_pdf_document)
    
    perf.assert_faster_than(60, "Document parsing")
    
    assert parse_result.text
    assert len(parse_result.chunks) > 0
    assert parse_result.metadata.get("doc_type")
    
    print(f"✓ Document parsed: {len(parse_result.text)} chars")
    print(f"✓ Chunks created: {len(parse_result.chunks)}")
    print(f"✓ Parsing method: {parse_result.parsing_method}")
    print(f"✓ Duration: {perf.duration:.2f}s")
    
    # ==================== STEP 2: Embedding & Indexing ====================
    
    print("\n=== STEP 2: Embedding & Indexing ===")
    
    collection_name = "test_workflow"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    # Generate embeddings
    chunk_texts = [chunk.text for chunk in parse_result.chunks]
    
    with measure_performance() as perf:
        embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    perf.assert_faster_than(30, "Embedding generation")
    
    assert len(embeddings) == len(chunk_texts)
    print(f"✓ Embeddings generated: {len(embeddings)} vectors")
    print(f"✓ Duration: {perf.duration:.2f}s")
    
    # Index in Qdrant
    metadata_list = [
        {
            "text": chunk.text,
            "chunk_id": chunk.chunk_id,
            "doc_id": "test_doc_1",
            **chunk.metadata,
        }
        for chunk in parse_result.chunks
    ]
    
    with measure_performance() as perf:
        indexed_count = await vector_store.upsert_vectors(
            collection_name,
            embeddings,
            metadata_list,
        )
    
    perf.assert_faster_than(10, "Vector indexing")
    
    assert indexed_count == len(embeddings)
    print(f"✓ Vectors indexed: {indexed_count}")
    print(f"✓ Duration: {perf.duration:.2f}s")
    
    # ==================== STEP 3: Query & Retrieval ====================
    
    print("\n=== STEP 3: Query & Retrieval ===")
    
    test_query = sample_queries[0]
    print(f"Query: {test_query}")
    
    # Validate query
    validated_query = validate_query(test_query)
    assert validated_query == test_query.strip()
    print(f"✓ Query validated")
    
    # Query engine
    async with RAGQueryEngine(collection_name) as engine:
        with measure_performance() as perf:
            query_result = await engine.query(
                validated_query,
                top_k=5,
            )
        
        perf.assert_faster_than(5, "Query retrieval")
        
        assert len(query_result.retrieved_docs) > 0
        assert query_result.query_embedding is not None
        
        print(f"✓ Retrieved documents: {len(query_result.retrieved_docs)}")
        print(f"✓ Top score: {query_result.retrieved_docs[0]['score']:.3f}")
        print(f"✓ Duration: {perf.duration:.2f}s")
    
    # ==================== STEP 4: Answer Generation ====================
    
    print("\n=== STEP 4: Answer Generation ===")
    
    with measure_performance() as perf:
        response = await response_synthesizer.synthesize(
            query=validated_query,
            retrieved_documents=query_result.retrieved_docs,
        )
    
    perf.assert_faster_than(10, "Answer generation")
    
    assert response.answer
    assert len(response.answer) > 20
    assert response.confidence > 0
    
    print(f"✓ Answer generated: {len(response.answer)} chars")
    print(f"✓ Confidence: {response.confidence:.3f}")
    print(f"✓ Citations: {len(response.citations)}")
    print(f"✓ Cost: ${response.llm_response.cost_usd:.4f}")
    print(f"✓ Duration: {perf.duration:.2f}s")
    print(f"\nAnswer preview:\n{response.answer[:200]}...")
    
    # ==================== STEP 5: Guardrails Check ====================
    
    print("\n=== STEP 5: Guardrails Check ===")
    
    guardrail_result = check_output(
        answer=response.answer,
        sources=[
            {"text": doc["metadata"]["text"], "score": doc["score"]}
            for doc in query_result.retrieved_docs
        ],
        confidence=response.confidence,
    )
    
    assert guardrail_result.safe_for_output or guardrail_result.needs_review
    
    print(f"✓ Status: {guardrail_result.status.value}")
    print(f"✓ Safe for output: {guardrail_result.safe_for_output}")
    print(f"✓ Needs review: {guardrail_result.needs_review}")
    print(f"✓ Issues: {len(guardrail_result.issues)}")
    print(f"✓ Warnings: {len(guardrail_result.warnings)}")
    
    if guardrail_result.issues:
        print(f"  Issues: {guardrail_result.issues}")
    if guardrail_result.warnings:
        print(f"  Warnings: {guardrail_result.warnings}")
    
    # ==================== STEP 6: RAGAS Evaluation ====================
    
    print("\n=== STEP 6: RAGAS Evaluation ===")
    
    # Create evaluation sample
    eval_sample = EvaluationSample(
        question=test_query,
        answer=response.answer,
        contexts=[doc["metadata"]["text"] for doc in query_result.retrieved_docs],
        ground_truth=sample_ground_truths[0],
    )
    
    with measure_performance() as perf:
        ragas_result = await ragas_evaluator.evaluate([eval_sample])
    
    perf.assert_faster_than(30, "RAGAS evaluation")
    
    # Assert minimum quality thresholds
    assert_metric_in_range(ragas_result.faithfulness, 0.6, 1.0, "Faithfulness")
    assert_metric_in_range(ragas_result.answer_relevancy, 0.5, 1.0, "Answer Relevancy")
    
    print(f"✓ Context Precision: {ragas_result.context_precision:.3f}")
    print(f"✓ Context Recall: {ragas_result.context_recall:.3f}")
    print(f"✓ Faithfulness: {ragas_result.faithfulness:.3f}")
    print(f"✓ Answer Relevancy: {ragas_result.answer_relevancy:.3f}")
    print(f"✓ Overall Score: {ragas_result.overall_score:.3f}")
    print(f"✓ Duration: {perf.duration:.2f}s")
    
    # ==================== STEP 7: Metrics Tracking ====================
    
    print("\n=== STEP 7: Metrics Tracking ===")
    
    # Record all metrics
    record_metric("context_precision", ragas_result.context_precision)
    record_metric("context_recall", ragas_result.context_recall)
    record_metric("faithfulness", ragas_result.faithfulness)
    record_metric("answer_relevancy", ragas_result.answer_relevancy)
    record_metric("overall_score", ragas_result.overall_score)
    record_metric("confidence", response.confidence)
    record_metric("response_time_seconds", perf.duration)
    
    print(f"✓ Metrics recorded: 7 metrics")
    
    # ==================== SUMMARY ====================
    
    print("\n" + "=" * 60)
    print("✅ E2E WORKFLOW TEST PASSED!")
    print("=" * 60)
    print(f"Document: {sample_pdf_document.name}")
    print(f"Chunks: {len(parse_result.chunks)}")
    print(f"Query: {test_query}")
    print(f"Answer length: {len(response.answer)} chars")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"RAGAS overall: {ragas_result.overall_score:.3f}")
    print(f"Guardrails: {guardrail_result.status.value}")
    print("=" * 60)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multiple_queries_workflow(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test workflow with multiple queries on same document."""
    
    print("\n=== Testing Multiple Queries ===")
    
    # Parse and index document
    parse_result = await parse_document(sample_pdf_document)
    
    collection_name = "test_multi_query"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in parse_result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    metadata_list = [
        {"text": chunk.text, "chunk_id": chunk.chunk_id}
        for chunk in parse_result.chunks
    ]
    
    await vector_store.upsert_vectors(
        collection_name,
        embeddings,
        metadata_list,
    )
    
    print(f"✓ Document indexed: {len(embeddings)} chunks")
    
    # Test each query
    async with RAGQueryEngine(collection_name) as engine:
        for i, query in enumerate(sample_queries[:3], 1):  # Test first 3
            print(f"\n--- Query {i}: {query}")
            
            # Retrieve
            query_result = await engine.query(query, top_k=3)
            assert len(query_result.retrieved_docs) > 0
            
            # Generate
            response = await response_synthesizer.synthesize(
                query=query,
                retrieved_documents=query_result.retrieved_docs,
            )
            
            assert response.answer
            assert response.confidence > 0
            
            print(f"  ✓ Answer: {len(response.answer)} chars")
            print(f"  ✓ Confidence: {response.confidence:.3f}")
            print(f"  ✓ Citations: {len(response.citations)}")
    
    print("\n✅ Multiple queries test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_error_recovery_workflow(
    sample_pdf_document: Path,
    vector_store,
):
    """Test error recovery in workflow."""
    
    print("\n=== Testing Error Recovery ===")
    
    # Test 1: Invalid file
    print("\n--- Test 1: Invalid file ---")
    with pytest.raises(Exception):
        validate_file(Path("/nonexistent/file.pdf"))
    print("✓ Invalid file handled correctly")
    
    # Test 2: Invalid query
    print("\n--- Test 2: Invalid query ---")
    from legal_assistant.core.exceptions import QueryValidationError
    with pytest.raises(QueryValidationError):
        validate_query("")  # Empty query
    print("✓ Invalid query handled correctly")
    
    # Test 3: Collection not found
    print("\n--- Test 3: Collection not found ---")
    async with RAGQueryEngine("nonexistent_collection") as engine:
        try:
            await engine.query("test query")
            assert False, "Should have raised error"
        except Exception as e:
            print(f"✓ Collection error handled: {type(e).__name__}")
    
    print("\n✅ Error recovery test passed!")
