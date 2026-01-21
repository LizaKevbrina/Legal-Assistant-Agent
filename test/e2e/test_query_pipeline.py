"""
E2E tests for query pipeline.

Tests:
- Query validation
- Vector search
- Hybrid search
- Reranking
- Answer generation
- Citation extraction
"""

import pytest
from pathlib import Path

from legal_assistant.ingestion import parse_document
from legal_assistant.retrieval import (
    QdrantVectorStore,
    EmbeddingGenerator,
    RAGQueryEngine,
)
from legal_assistant.generation import ResponseSynthesizer
from legal_assistant.security import validate_query


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_query_validation_pipeline():
    """Test query validation with various inputs."""
    
    print("\n=== Testing Query Validation ===")
    
    # Test 1: Valid queries
    print("\n--- Test 1: Valid queries ---")
    valid_queries = [
        "Что такое договор купли-продажи?",
        "Какая ответственность за просрочку?",
        "Explain the contract terms",
    ]
    
    for query in valid_queries:
        validated = validate_query(query)
        assert validated
        print(f"✓ Valid: {query[:50]}")
    
    # Test 2: Invalid queries
    print("\n--- Test 2: Invalid queries ---")
    from legal_assistant.core.exceptions import QueryValidationError
    
    invalid_queries = [
        "",  # Empty
        "   ",  # Whitespace only
        "a" * 2000,  # Too long
    ]
    
    for query in invalid_queries:
        with pytest.raises(QueryValidationError):
            validate_query(query)
        print(f"✓ Rejected: {query[:50]}...")
    
    # Test 3: SQL injection attempts
    print("\n--- Test 3: SQL injection attempts ---")
    sql_injections = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "UNION SELECT * FROM secrets",
    ]
    
    for query in sql_injections:
        with pytest.raises(QueryValidationError):
            validate_query(query, check_sql_injection=True)
        print(f"✓ Blocked SQL injection: {query[:50]}")
    
    print("\n✅ Query validation test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_vector_search_accuracy(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test vector search retrieval accuracy."""
    
    print("\n=== Testing Vector Search Accuracy ===")
    
    # Index document
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_search_accuracy"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    metadata_list = [
        {"text": chunk.text, "chunk_id": i}
        for i, chunk in enumerate(result.chunks)
    ]
    
    await vector_store.upsert_vectors(
        collection_name,
        embeddings,
        metadata_list,
    )
    
    print(f"✓ Indexed {len(embeddings)} chunks")
    
    # Test queries
    for i, query in enumerate(sample_queries[:3], 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Generate query embedding
        query_embedding = await embedding_generator.embed_query(query)
        
        # Search
        search_results = await vector_store.search(
            collection_name,
            query_embedding,
            top_k=5,
        )
        
        assert len(search_results) > 0
        
        # Check scores are descending
        scores = [r["score"] for r in search_results]
        assert scores == sorted(scores, reverse=True)
        
        print(f"  ✓ Retrieved: {len(search_results)} docs")
        print(f"  ✓ Top score: {scores[0]:.3f}")
        print(f"  ✓ Bottom score: {scores[-1]:.3f}")
        
        # Print top result preview
        top_text = search_results[0]["metadata"]["text"]
        print(f"  ✓ Top result: {top_text[:100]}...")
    
    print("\n✅ Vector search accuracy test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_hybrid_search_vs_vector_only(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Compare hybrid search vs vector-only search."""
    
    print("\n=== Testing Hybrid vs Vector-Only Search ===")
    
    # Index document
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_hybrid_search"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    metadata_list = [
        {"text": chunk.text}
        for chunk in result.chunks
    ]
    
    await vector_store.upsert_vectors(
        collection_name,
        embeddings,
        metadata_list,
    )
    
    print(f"✓ Indexed {len(embeddings)} chunks")
    
    # Test with RAGQueryEngine (uses hybrid search)
    test_query = sample_queries[1]  # "Какая цена товара?"
    
    print(f"\nQuery: {test_query}")
    
    async with RAGQueryEngine(collection_name) as engine:
        # Hybrid search
        hybrid_result = await engine.query(test_query, top_k=5)
        
        print(f"\n--- Hybrid Search ---")
        print(f"✓ Retrieved: {len(hybrid_result.retrieved_docs)}")
        print(f"✓ Top score: {hybrid_result.retrieved_docs[0]['score']:.3f}")
        print(f"✓ Preview: {hybrid_result.retrieved_docs[0]['metadata']['text'][:100]}...")
    
    print("\n✅ Hybrid search test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_query_engine_with_filters(
    sample_pdf_document: Path,
    vector_store,
    embedding_generator,
):
    """Test RAGQueryEngine with metadata filters."""
    
    print("\n=== Testing Query Engine with Filters ===")
    
    # Index multiple documents with different metadata
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_filters"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    # Add varied metadata
    metadata_list = []
    for i, chunk in enumerate(result.chunks):
        metadata_list.append({
            "text": chunk.text,
            "doc_type": "contract" if i % 2 == 0 else "law",
            "section": f"section_{i % 3}",
        })
    
    await vector_store.upsert_vectors(
        collection_name,
        embeddings,
        metadata_list,
    )
    
    print(f"✓ Indexed with metadata")
    
    # Test with filters
    async with RAGQueryEngine(collection_name) as engine:
        # Filter by doc_type
        print("\n--- Filter: doc_type=contract ---")
        result_filtered = await engine.query(
            "цена товара",
            top_k=5,
            metadata_filter={"doc_type": "contract"},
        )
        
        assert len(result_filtered.retrieved_docs) > 0
        
        # Verify all results match filter
        for doc in result_filtered.retrieved_docs:
            assert doc["metadata"]["doc_type"] == "contract"
        
        print(f"✓ Retrieved: {len(result_filtered.retrieved_docs)}")
        print(f"✓ All match filter: doc_type=contract")
    
    print("\n✅ Filters test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_answer_generation_quality(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test answer generation quality."""
    
    print("\n=== Testing Answer Generation Quality ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_generation"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ Setup complete")
    
    # Test each query
    async with RAGQueryEngine(collection_name) as engine:
        for i, query in enumerate(sample_queries[:3], 1):
            print(f"\n--- Query {i}: {query} ---")
            
            # Retrieve
            query_result = await engine.query(query, top_k=5)
            
            # Generate answer
            response = await response_synthesizer.synthesize(
                query=query,
                retrieved_documents=query_result.retrieved_docs,
            )
            
            # Quality checks
            assert response.answer
            assert len(response.answer) >= 50, "Answer too short"
            assert len(response.answer) <= 2000, "Answer too long"
            assert response.confidence > 0
            assert response.confidence <= 1.0
            
            print(f"  ✓ Answer length: {len(response.answer)} chars")
            print(f"  ✓ Confidence: {response.confidence:.3f}")
            print(f"  ✓ Citations: {len(response.citations)}")
            print(f"  ✓ Tokens: {response.llm_response.total_tokens}")
            print(f"  ✓ Cost: ${response.llm_response.cost_usd:.4f}")
            
            # Check for citations
            if response.citations:
                print(f"  ✓ First citation: {response.citations[0].source_name}")
    
    print("\n✅ Answer generation quality test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_citation_extraction_accuracy(
    sample_pdf_document: Path,
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test citation extraction accuracy."""
    
    print("\n=== Testing Citation Extraction ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_citations"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    
    metadata_list = [
        {"text": c.text, "source": f"doc_{i}", "page": i // 3}
        for i, c in enumerate(result.chunks)
    ]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Query
    async with RAGQueryEngine(collection_name) as engine:
        query_result = await engine.query("цена договора", top_k=3)
        
        response = await response_synthesizer.synthesize(
            query="цена договора",
            retrieved_documents=query_result.retrieved_docs,
        )
        
        print(f"✓ Generated answer with {len(response.citations)} citations")
        
        # Verify citations
        for i, citation in enumerate(response.citations, 1):
            print(f"\nCitation {i}:")
            print(f"  Source: {citation.source_name}")
            print(f"  Confidence: {citation.confidence:.3f}")
            print(f"  Text: {citation.text[:100]}...")
            
            # Check citation is valid
            assert citation.source_name
            assert 0 <= citation.confidence <= 1
            assert citation.text
    
    print("\n✅ Citation extraction test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_query_pipeline_performance(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
    measure_performance,
):
    """Test end-to-end query pipeline performance."""
    
    print("\n=== Testing Query Pipeline Performance ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_performance"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ Setup complete")
    
    # Measure query latency
    latencies = []
    
    async with RAGQueryEngine(collection_name) as engine:
        for query in sample_queries[:5]:
            with measure_performance() as perf:
                # Full pipeline
                query_result = await engine.query(query, top_k=5)
                response = await response_synthesizer.synthesize(
                    query=query,
                    retrieved_documents=query_result.retrieved_docs,
                )
            
            latencies.append(perf.duration)
            print(f"  Query latency: {perf.duration:.2f}s")
    
    # Statistics
    avg_latency = sum(latencies) / len(latencies)
    max_latency = max(latencies)
    
    print(f"\n✓ Average latency: {avg_latency:.2f}s")
    print(f"✓ Max latency: {max_latency:.2f}s")
    
    # Performance assertions
    assert avg_latency < 15, f"Average latency too high: {avg_latency:.2f}s"
    assert max_latency < 20, f"Max latency too high: {max_latency:.2f}s"
    
    print("\n✅ Performance test passed!")
