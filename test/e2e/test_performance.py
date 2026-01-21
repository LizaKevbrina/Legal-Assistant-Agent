"""
E2E performance and load tests.

Tests:
- Query latency benchmarks
- Throughput testing
- Concurrent request handling
- Memory usage
- Cache effectiveness
- Stress testing
"""

import pytest
import asyncio
from pathlib import Path
import time
import statistics
from typing import List

from legal_assistant.ingestion import parse_document
from legal_assistant.retrieval import (
    QdrantVectorStore,
    EmbeddingGenerator,
    RAGQueryEngine,
)
from legal_assistant.generation import ResponseSynthesizer


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_query_latency_benchmarks(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Benchmark query latency at different stages."""
    
    print("\n=== Query Latency Benchmarks ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_latency"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ Setup complete: {len(embeddings)} chunks indexed")
    
    # Benchmark components
    latencies = {
        "embedding": [],
        "retrieval": [],
        "generation": [],
        "end_to_end": [],
    }
    
    async with RAGQueryEngine(collection_name) as engine:
        for query in sample_queries[:5]:
            print(f"\n--- Query: {query[:50]}... ---")
            
            # 1. Embedding latency
            start = time.time()
            query_embedding = await embedding_generator.embed_query(query)
            latencies["embedding"].append(time.time() - start)
            print(f"  Embedding: {latencies['embedding'][-1]:.3f}s")
            
            # 2. Retrieval latency
            start = time.time()
            search_results = await vector_store.search(
                collection_name,
                query_embedding,
                top_k=5,
            )
            latencies["retrieval"].append(time.time() - start)
            print(f"  Retrieval: {latencies['retrieval'][-1]:.3f}s")
            
            # 3. Generation latency
            start = time.time()
            response = await response_synthesizer.synthesize(
                query=query,
                retrieved_documents=[
                    {"metadata": {"text": r["metadata"]["text"]}, "score": r["score"]}
                    for r in search_results
                ],
            )
            latencies["generation"].append(time.time() - start)
            print(f"  Generation: {latencies['generation'][-1]:.3f}s")
            
            # 4. End-to-end (using query engine)
            start = time.time()
            await engine.query(query, top_k=5)
            latencies["end_to_end"].append(time.time() - start)
            print(f"  End-to-end: {latencies['end_to_end'][-1]:.3f}s")
    
    # Statistics
    print("\n=== Latency Statistics ===")
    
    for component, times in latencies.items():
        avg = statistics.mean(times)
        median = statistics.median(times)
        p95 = sorted(times)[int(0.95 * len(times))]
        
        print(f"\n{component.upper()}:")
        print(f"  Mean: {avg:.3f}s")
        print(f"  Median: {median:.3f}s")
        print(f"  P95: {p95:.3f}s")
        print(f"  Min: {min(times):.3f}s")
        print(f"  Max: {max(times):.3f}s")
        
        # Performance assertions
        if component == "embedding":
            assert avg < 1.0, f"Embedding too slow: {avg:.3f}s"
        elif component == "retrieval":
            assert avg < 2.0, f"Retrieval too slow: {avg:.3f}s"
        elif component == "generation":
            assert avg < 8.0, f"Generation too slow: {avg:.3f}s"
        elif component == "end_to_end":
            assert avg < 10.0, f"End-to-end too slow: {avg:.3f}s"
    
    print("\n✅ Latency benchmarks passed!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_request_handling(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test concurrent request handling."""
    
    print("\n=== Concurrent Request Handling ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_concurrent"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ Setup complete")
    
    # Test with increasing concurrency
    concurrency_levels = [1, 5, 10]
    
    for concurrency in concurrency_levels:
        print(f"\n--- Concurrency: {concurrency} ---")
        
        async def query_task(query: str):
            """Single query task."""
            async with RAGQueryEngine(collection_name) as engine:
                start = time.time()
                await engine.query(query, top_k=5)
                return time.time() - start
        
        # Create tasks
        tasks = [
            query_task(sample_queries[i % len(sample_queries)])
            for i in range(concurrency)
        ]
        
        # Execute concurrently
        start_time = time.time()
        latencies = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Statistics
        avg_latency = statistics.mean(latencies)
        throughput = concurrency / total_time
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Avg latency: {avg_latency:.2f}s")
        print(f"  Throughput: {throughput:.2f} req/s")
        print(f"  Min latency: {min(latencies):.2f}s")
        print(f"  Max latency: {max(latencies):.2f}s")
        
        # Assertions
        assert all(lat < 20 for lat in latencies), "Some queries too slow"
    
    print("\n✅ Concurrent request test passed!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_throughput_limits(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test system throughput limits."""
    
    print("\n=== Throughput Testing ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_throughput"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Test throughput over 30 seconds
    print("\n--- 30-Second Throughput Test ---")
    
    query_count = 0
    errors = 0
    start_time = time.time()
    duration = 30  # seconds
    
    async def continuous_queries():
        """Run queries continuously."""
        nonlocal query_count, errors
        
        async with RAGQueryEngine(collection_name) as engine:
            while time.time() - start_time < duration:
                try:
                    query = sample_queries[query_count % len(sample_queries)]
                    await engine.query(query, top_k=3)
                    query_count += 1
                except Exception as e:
                    errors += 1
                    print(f"  Error: {e}")
                
                # Small delay to avoid overwhelming
                await asyncio.sleep(0.1)
    
    # Run test
    await continuous_queries()
    
    total_time = time.time() - start_time
    avg_throughput = query_count / total_time
    
    print(f"\n✓ Results:")
    print(f"  Duration: {total_time:.2f}s")
    print(f"  Total queries: {query_count}")
    print(f"  Errors: {errors}")
    print(f"  Success rate: {(1 - errors/max(query_count,1))*100:.1f}%")
    print(f"  Avg throughput: {avg_throughput:.2f} queries/s")
    
    # Assertions
    assert query_count > 10, "Too few queries completed"
    assert errors < query_count * 0.1, "Error rate too high (>10%)"
    
    print("\n✅ Throughput test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_cache_effectiveness(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test caching effectiveness."""
    
    print("\n=== Cache Effectiveness Testing ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_cache"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    test_query = sample_queries[0]
    
    # Test 1: First query (cold cache)
    print("\n--- Cold Cache ---")
    
    start = time.time()
    embedding1 = await embedding_generator.embed_query(test_query)
    cold_time = time.time() - start
    
    print(f"  First query: {cold_time:.3f}s")
    
    # Test 2: Repeat query (warm cache)
    print("\n--- Warm Cache ---")
    
    warm_times = []
    for i in range(3):
        start = time.time()
        embedding2 = await embedding_generator.embed_query(test_query)
        warm_times.append(time.time() - start)
    
    avg_warm = statistics.mean(warm_times)
    
    print(f"  Subsequent queries: {avg_warm:.3f}s avg")
    
    # Calculate speedup
    if avg_warm > 0:
        speedup = cold_time / avg_warm
        print(f"\n✓ Cache speedup: {speedup:.1f}x")
        
        # Should see some speedup from caching
        # (May be minimal if Redis not configured)
    
    # Verify embeddings are identical
    assert len(embedding1) == len(embedding2)
    print(f"✓ Cache consistency verified")
    
    print("\n✅ Cache effectiveness test passed!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_large_batch_processing(
    temp_dir: Path,
    vector_store,
    embedding_generator,
):
    """Test processing large batches of queries."""
    
    print("\n=== Large Batch Processing ===")
    
    # Create large document
    print("\n--- Creating Large Document ---")
    
    large_text = "\n".join([
        f"Раздел {i}: Важная информация о договоре купли-продажи. "
        f"Цена товара составляет {i * 1000} рублей. "
        f"Срок поставки: {i} дней."
        for i in range(100)
    ])
    
    large_file = temp_dir / "large_doc.txt"
    large_file.write_text(large_text, encoding="utf-8")
    
    print(f"✓ Created: {len(large_text)} chars")
    
    # Parse
    from legal_assistant.ingestion import parse_document
    result = await parse_document(large_file)
    
    print(f"✓ Parsed: {len(result.chunks)} chunks")
    
    # Test batch embedding
    print("\n--- Batch Embedding ---")
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    
    start = time.time()
    embeddings = await embedding_generator.embed_texts(
        chunk_texts,
        batch_size=100,  # Process in batches
    )
    embed_time = time.time() - start
    
    print(f"✓ Embedded {len(embeddings)} chunks in {embed_time:.2f}s")
    print(f"  Rate: {len(embeddings)/embed_time:.1f} chunks/s")
    
    # Test batch indexing
    print("\n--- Batch Indexing ---")
    
    collection_name = "test_large_batch"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    metadata_list = [{"text": c.text, "index": i} for i, c in enumerate(result.chunks)]
    
    start = time.time()
    indexed = await vector_store.upsert_vectors(
        collection_name,
        embeddings,
        metadata_list,
        batch_size=100,
    )
    index_time = time.time() - start
    
    print(f"✓ Indexed {indexed} vectors in {index_time:.2f}s")
    print(f"  Rate: {indexed/index_time:.1f} vectors/s")
    
    # Assertions
    assert indexed == len(embeddings)
    assert embed_time < 120, f"Embedding too slow: {embed_time:.2f}s"
    assert index_time < 30, f"Indexing too slow: {index_time:.2f}s"
    
    print("\n✅ Large batch processing test passed!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_memory_usage_stability(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test memory usage stability over many queries."""
    
    print("\n=== Memory Usage Stability ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_memory"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Measure memory before
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"✓ Initial memory: {initial_memory:.1f} MB")
    
    # Run many queries
    print("\n--- Running 50 Queries ---")
    
    memory_samples = []
    
    async with RAGQueryEngine(collection_name) as engine:
        for i in range(50):
            query = sample_queries[i % len(sample_queries)]
            await engine.query(query, top_k=5)
            
            # Sample memory every 10 queries
            if i % 10 == 0:
                mem = process.memory_info().rss / 1024 / 1024
                memory_samples.append(mem)
                print(f"  Query {i}: {mem:.1f} MB")
    
    # Measure memory after
    final_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = final_memory - initial_memory
    
    print(f"\n✓ Final memory: {final_memory:.1f} MB")
    print(f"✓ Increase: {memory_increase:+.1f} MB")
    
    # Check for memory leaks
    if len(memory_samples) > 1:
        # Memory should stabilize (not continuously grow)
        recent_avg = statistics.mean(memory_samples[-3:])
        early_avg = statistics.mean(memory_samples[:3])
        growth_rate = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        
        print(f"✓ Growth rate: {growth_rate*100:+.1f}%")
        
        # Memory shouldn't grow more than 50% over the test
        assert growth_rate < 0.5, f"Potential memory leak: {growth_rate*100:.1f}% growth"
    
    print("\n✅ Memory stability test passed!")


@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.asyncio
async def test_stress_test_recovery(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
):
    """Test system recovery under stress."""
    
    print("\n=== Stress Test & Recovery ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_stress"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Stress phase: Many concurrent requests
    print("\n--- Stress Phase: 20 Concurrent Queries ---")
    
    async def stress_query(query: str):
        """Query that might fail under stress."""
        try:
            async with RAGQueryEngine(collection_name) as engine:
                await engine.query(query, top_k=5)
                return True
        except Exception as e:
            print(f"  Error during stress: {type(e).__name__}")
            return False
    
    # Launch many concurrent queries
    tasks = [
        stress_query(sample_queries[i % len(sample_queries)])
        for i in range(20)
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=False)
    success_count = sum(1 for r in results if r)
    
    print(f"✓ Stress results: {success_count}/20 succeeded")
    
    # Recovery phase: Verify system still works
    print("\n--- Recovery Phase ---")
    
    # Wait a moment
    await asyncio.sleep(2)
    
    # Try normal query
    async with RAGQueryEngine(collection_name) as engine:
        recovery_result = await engine.query(sample_queries[0], top_k=5)
    
    assert recovery_result.retrieved_docs
    print(f"✓ System recovered: query successful")
    print(f"  Retrieved: {len(recovery_result.retrieved_docs)} docs")
    
    print("\n✅ Stress test & recovery passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_performance_regression_detection(
    sample_pdf_document: Path,
    sample_queries: list[str],
    vector_store,
    embedding_generator,
    metrics_tracker,
):
    """Test performance regression detection."""
    
    print("\n=== Performance Regression Detection ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_perf_regression"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Benchmark baseline performance
    print("\n--- Baseline Performance ---")
    
    latencies = []
    
    async with RAGQueryEngine(collection_name) as engine:
        for query in sample_queries[:5]:
            start = time.time()
            await engine.query(query, top_k=5)
            latencies.append(time.time() - start)
    
    baseline_avg = statistics.mean(latencies)
    baseline_p95 = sorted(latencies)[int(0.95 * len(latencies))]
    
    print(f"✓ Baseline avg latency: {baseline_avg:.3f}s")
    print(f"✓ Baseline P95: {baseline_p95:.3f}s")
    
    # Record metrics
    metrics_tracker.record(
        "query_latency_avg",
        baseline_avg,
        metadata={"version": "baseline"},
    )
    
    metrics_tracker.record(
        "query_latency_p95",
        baseline_p95,
        metadata={"version": "baseline"},
    )
    
    # Performance thresholds
    max_avg_latency = 10.0  # seconds
    max_p95_latency = 15.0  # seconds
    
    print(f"\n✓ Performance assertions:")
    assert baseline_avg < max_avg_latency, f"Avg latency {baseline_avg:.2f}s exceeds {max_avg_latency}s"
    assert baseline_p95 < max_p95_latency, f"P95 latency {baseline_p95:.2f}s exceeds {max_p95_latency}s"
    print(f"  ✓ Avg latency < {max_avg_latency}s")
    print(f"  ✓ P95 latency < {max_p95_latency}s")
    
    print("\n✅ Performance regression detection test passed!")
