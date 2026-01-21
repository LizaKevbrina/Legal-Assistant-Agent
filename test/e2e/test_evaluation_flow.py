"""
E2E tests for evaluation flow.

Tests:
- RAGAS evaluation pipeline
- LLM-as-Judge evaluation
- Dataset management workflow
- Metrics tracking
- Regression detection
"""

import pytest
from pathlib import Path

from legal_assistant.ingestion import parse_document
from legal_assistant.retrieval import RAGQueryEngine
from legal_assistant.generation import ResponseSynthesizer
from legal_assistant.evaluation import (
    RAGASEvaluator,
    LLMJudge,
    DatasetManager,
    MetricsTracker,
    RegressionTestSuite,
    EvaluationSample,
    DatasetSample,
    JudgmentCriterion,
    SplitType,
)


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.slow
async def test_ragas_evaluation_pipeline(
    sample_pdf_document: Path,
    sample_queries: list[str],
    sample_ground_truths: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test complete RAGAS evaluation pipeline."""
    
    print("\n=== Testing RAGAS Evaluation Pipeline ===")
    
    # Setup RAG system
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_ragas"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    print(f"✓ RAG system ready")
    
    # Generate Q&A pairs
    print("\n--- Generating Q&A Pairs ---")
    
    eval_samples = []
    
    async with RAGQueryEngine(collection_name) as engine:
        for i, (query, ground_truth) in enumerate(
            zip(sample_queries[:3], sample_ground_truths[:3]), 1
        ):
            print(f"\nPair {i}: {query}")
            
            # Retrieve
            query_result = await engine.query(query, top_k=5)
            
            # Generate
            response = await response_synthesizer.synthesize(
                query=query,
                retrieved_documents=query_result.retrieved_docs,
            )
            
            # Create evaluation sample
            sample = EvaluationSample(
                question=query,
                answer=response.answer,
                contexts=[
                    doc["metadata"]["text"]
                    for doc in query_result.retrieved_docs
                ],
                ground_truth=ground_truth,
            )
            
            eval_samples.append(sample)
            
            print(f"  ✓ Answer: {len(response.answer)} chars")
            print(f"  ✓ Contexts: {len(sample.contexts)}")
    
    # Run RAGAS evaluation
    print("\n--- RAGAS Evaluation ---")
    
    evaluator = RAGASEvaluator()
    ragas_result = await evaluator.evaluate(eval_samples)
    
    print(ragas_result.get_summary())
    
    # Assertions
    assert ragas_result.sample_count == len(eval_samples)
    assert 0 <= ragas_result.context_precision <= 1
    assert 0 <= ragas_result.context_recall <= 1
    assert 0 <= ragas_result.faithfulness <= 1
    assert 0 <= ragas_result.answer_relevancy <= 1
    
    # Quality thresholds
    assert ragas_result.faithfulness > 0.5, "Faithfulness too low"
    assert ragas_result.overall_score > 0.4, "Overall score too low"
    
    print("\n✅ RAGAS evaluation pipeline test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
@pytest.mark.slow
async def test_llm_judge_evaluation(
    sample_pdf_document: Path,
    sample_queries: list[str],
    sample_ground_truths: list[str],
    vector_store,
    embedding_generator,
    response_synthesizer,
):
    """Test LLM-as-Judge evaluation."""
    
    print("\n=== Testing LLM Judge Evaluation ===")
    
    # Setup
    result = await parse_document(sample_pdf_document)
    
    collection_name = "test_llm_judge"
    await vector_store.create_collection(collection_name, vector_size=3072)
    
    chunk_texts = [chunk.text for chunk in result.chunks]
    embeddings = await embedding_generator.embed_texts(chunk_texts)
    metadata_list = [{"text": c.text} for c in result.chunks]
    await vector_store.upsert_vectors(collection_name, embeddings, metadata_list)
    
    # Generate answer
    async with RAGQueryEngine(collection_name) as engine:
        query = sample_queries[0]
        ground_truth = sample_ground_truths[0]
        
        query_result = await engine.query(query, top_k=5)
        response = await response_synthesizer.synthesize(
            query=query,
            retrieved_documents=query_result.retrieved_docs,
        )
    
    print(f"✓ Answer generated for evaluation")
    
    # LLM Judge evaluation
    print("\n--- LLM Judge Evaluation ---")
    
    judge = LLMJudge(model="gpt-4-turbo")
    
    judgment = await judge.evaluate(
        question=query,
        answer=response.answer,
        reference=ground_truth,
        contexts=[doc["metadata"]["text"] for doc in query_result.retrieved_docs],
        criteria=[
            JudgmentCriterion.CORRECTNESS,
            JudgmentCriterion.COMPLETENESS,
            JudgmentCriterion.CLARITY,
        ],
    )
    
    print(judgment.get_summary())
    
    # Assertions
    assert 0 <= judgment.overall_score <= 1
    assert judgment.verdict in ["excellent", "good", "acceptable", "poor"]
    assert len(judgment.scores) >= 3
    assert judgment.summary
    
    # Check individual criteria
    for score in judgment.scores:
        print(f"\n{score.criterion.value}:")
        print(f"  Score: {score.score:.3f}")
        print(f"  Explanation: {score.explanation[:100]}...")
        
        assert 0 <= score.score <= 1
        assert score.explanation
    
    print("\n✅ LLM Judge evaluation test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_dataset_management_workflow(
    sample_queries: list[str],
    sample_ground_truths: list[str],
    dataset_manager: DatasetManager,
):
    """Test dataset management workflow."""
    
    print("\n=== Testing Dataset Management Workflow ===")
    
    # Create dataset samples
    print("\n--- Creating Dataset ---")
    
    samples = []
    for i, (query, truth) in enumerate(zip(sample_queries, sample_ground_truths)):
        sample = DatasetSample(
            id=f"sample_{i}",
            question=query,
            answer=truth,
            contexts=[f"Context {i}"],
            metadata={"difficulty": "medium", "topic": "contract_law"},
        )
        samples.append(sample)
    
    print(f"✓ Created {len(samples)} samples")
    
    # Save dataset
    print("\n--- Saving Dataset ---")
    
    metadata = dataset_manager.save_dataset(
        name="legal_qa_test",
        samples=samples,
        description="Test legal Q&A dataset",
        tags=["contract", "test"],
    )
    
    print(f"✓ Dataset saved:")
    print(f"  Name: {metadata.name}")
    print(f"  Version: {metadata.version}")
    print(f"  Samples: {metadata.sample_count}")
    print(f"  Checksum: {metadata.checksum}")
    
    # Load dataset
    print("\n--- Loading Dataset ---")
    
    loaded_metadata, loaded_samples = dataset_manager.load_dataset(
        "legal_qa_test",
        version=metadata.version,
    )
    
    assert len(loaded_samples) == len(samples)
    assert loaded_metadata.checksum == metadata.checksum
    
    print(f"✓ Dataset loaded: {len(loaded_samples)} samples")
    
    # Create splits
    print("\n--- Creating Splits ---")
    
    splits = dataset_manager.create_splits(
        loaded_samples,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    
    print(f"✓ Splits created:")
    print(f"  Train: {len(splits[SplitType.TRAIN])} samples")
    print(f"  Val: {len(splits[SplitType.VAL])} samples")
    print(f"  Test: {len(splits[SplitType.TEST])} samples")
    
    # Validate splits sum to total
    total_split = sum(len(s) for s in splits.values())
    assert total_split == len(loaded_samples)
    
    # Export dataset
    print("\n--- Exporting Dataset ---")
    
    export_path = dataset_manager.export_dataset(
        "legal_qa_test",
        format="jsonl",
    )
    
    assert export_path.exists()
    print(f"✓ Exported to: {export_path}")
    
    # Validate dataset
    print("\n--- Validating Dataset ---")
    
    validation_report = dataset_manager.validate_dataset(loaded_samples)
    
    print(f"✓ Validation report:")
    print(f"  Valid: {validation_report['valid']}")
    print(f"  Issues: {len(validation_report['issues'])}")
    print(f"  Warnings: {len(validation_report['warnings'])}")
    
    if validation_report['issues']:
        for issue in validation_report['issues']:
            print(f"    - {issue}")
    
    print("\n✅ Dataset management workflow test passed!")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_metrics_tracking_and_trending(
    metrics_tracker: MetricsTracker,
):
    """Test metrics tracking and trend analysis."""
    
    print("\n=== Testing Metrics Tracking & Trending ===")
    
    # Record metrics over time
    print("\n--- Recording Metrics ---")
    
    import random
    from datetime import datetime, timedelta
    
    base_date = datetime.utcnow() - timedelta(days=30)
    
    # Simulate 30 days of metrics
    for day in range(30):
        date = base_date + timedelta(days=day)
        
        # Simulate degradation in week 3
        if 14 <= day < 21:
            faithfulness = 0.7 + random.uniform(-0.1, 0.1)
        else:
            faithfulness = 0.85 + random.uniform(-0.05, 0.05)
        
        metrics_tracker.record(
            "faithfulness",
            faithfulness,
            timestamp=date,
            metadata={"day": day},
        )
    
    print(f"✓ Recorded 30 days of metrics")
    
    # Get statistics
    print

stats = metrics_tracker.get_statistics("faithfulness")

print(f"✓ Faithfulness statistics:")
print(f"  Count: {stats.count}")
print(f"  Mean: {stats.mean:.3f}")
print(f"  Median: {stats.median:.3f}")
print(f"  Std Dev: {stats.std_dev:.3f}")
print(f"  Min: {stats.min_value:.3f}")
print(f"  Max: {stats.max_value:.3f}")
print(f"  P95: {stats.p95:.3f}")
print(f"  P99: {stats.p99:.3f}")

# Analyze trend
print("\n--- Trend Analysis ---")

trend = metrics_tracker.analyze_trend(
    "faithfulness",
    recent_days=7,
    baseline_days=30,
)

if trend:
    print(f"✓ Trend analysis:")
    print(f"  Trend: {trend.trend}")
    print(f"  Change: {trend.change_percentage:+.1f}%")
    print(f"  Recent mean: {trend.recent_mean:.3f}")
    print(f"  Baseline mean: {trend.baseline_mean:.3f}")
    print(f"  Confidence: {trend.confidence:.3f}")

# Check alerts
print("\n--- Alert Checking ---")

alerts = metrics_tracker.check_alerts({
    "faithfulness": {"min": 0.75},
})

if alerts:
    print(f"⚠️  Alerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"    {alert['message']}")
else:
    print(f"✓ No alerts")

# Export for dashboard
print("\n--- Dashboard Export ---")

dashboard_data = metrics_tracker.export_for_dashboard(
    metrics=["faithfulness"],
    days=30,
)

print(f"✓ Dashboard data exported:")
print(f"  Metrics: {len(dashboard_data['metrics'])}")
print(f"  Period: {dashboard_data['period_days']} days")

print("\n✅ Metrics tracking test passed!")
