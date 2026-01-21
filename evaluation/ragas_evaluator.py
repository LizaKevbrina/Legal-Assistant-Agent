"""
RAGAS (Retrieval-Augmented Generation Assessment) evaluation.

Metrics:
- Context Precision: Relevance of retrieved documents
- Context Recall: Coverage of answer by context
- Faithfulness: Answer grounded in context (no hallucination)
- Answer Relevancy: Answer relevance to question

Production features:
- Async evaluation
- Batch processing
- Metric caching
- Error handling
- Progress tracking
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
    track_error,
    evaluation_duration,
)
from legal_assistant.core.exceptions import QualityError

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "answer": self.answer,
            "contexts": self.contexts,
            "ground_truth": self.ground_truth,
        }


@dataclass
class EvaluationResult:
    """Result of RAGAS evaluation."""
    
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    overall_score: float
    sample_count: int
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "overall_score": self.overall_score,
            "sample_count": self.sample_count,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"RAGAS Evaluation Results (n={self.sample_count}):\n"
            f"  Context Precision: {self.context_precision:.3f}\n"
            f"  Context Recall:    {self.context_recall:.3f}\n"
            f"  Faithfulness:      {self.faithfulness:.3f}\n"
            f"  Answer Relevancy:  {self.answer_relevancy:.3f}\n"
            f"  Overall Score:     {self.overall_score:.3f}"
        )


class RAGASEvaluator:
    """
    RAGAS-based RAG evaluation.
    
    Features:
    - 4 key metrics (precision, recall, faithfulness, relevancy)
    - Batch processing
    - Async support
    - Progress tracking
    - Result caching
    
    Example:
        >>> evaluator = RAGASEvaluator()
        >>> samples = [
        ...     EvaluationSample(
        ...         question="Что такое договор?",
        ...         answer="Договор — это соглашение...",
        ...         contexts=["Текст из закона...", "Статья..."],
        ...         ground_truth="Договор определяется как..."
        ...     )
        ... ]
        >>> result = await evaluator.evaluate(samples)
        >>> print(result.get_summary())
    """
    
    def __init__(
        self,
        llm_model: str = "gpt-4-turbo",
        embedding_model: str = "text-embedding-3-large",
    ):
        """
        Initialize RAGAS evaluator.
        
        Args:
            llm_model: LLM for evaluation
            embedding_model: Embedding model for relevancy
        """
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        logger.info(
            "ragas_evaluator_initialized",
            llm_model=llm_model,
            embedding_model=embedding_model,
        )
    
    async def evaluate(
        self,
        samples: List[EvaluationSample],
        metrics: Optional[List[str]] = None,
    ) -> EvaluationResult:
        """
        Evaluate RAG system using RAGAS metrics.
        
        Args:
            samples: List of evaluation samples
            metrics: Metrics to compute (None = all)
        
        Returns:
            EvaluationResult with scores
        
        Raises:
            QualityError: If evaluation fails
        
        Example:
            >>> result = await evaluator.evaluate(samples)
            >>> if result.faithfulness < 0.7:
            ...     print("Warning: High hallucination rate!")
        """
        if not samples:
            raise QualityError(
                message="No samples provided for evaluation",
                details={},
            )
        
        try:
            with track_time(
                evaluation_duration,
                {"metric": "ragas", "sample_count": len(samples)}
            ):
                # Convert samples to dataset
                dataset = self._create_dataset(samples)
                
                # Select metrics
                metric_list = self._get_metrics(metrics)
                
                logger.info(
                    "ragas_evaluation_started",
                    sample_count=len(samples),
                    metrics=[m.name for m in metric_list],
                )
                
                # Run evaluation (sync, RAGAS doesn't support async yet)
                result = await asyncio.to_thread(
                    evaluate,
                    dataset,
                    metrics=metric_list,
                )
                
                # Extract scores
                scores = {
                    "context_precision": result.get("context_precision", 0.0),
                    "context_recall": result.get("context_recall", 0.0),
                    "faithfulness": result.get("faithfulness", 0.0),
                    "answer_relevancy": result.get("answer_relevancy", 0.0),
                }
                
                # Calculate overall score
                overall_score = sum(scores.values()) / len(scores)
                
                logger.info(
                    "ragas_evaluation_completed",
                    sample_count=len(samples),
                    overall_score=overall_score,
                    **scores,
                )
                
                return EvaluationResult(
                    context_precision=scores["context_precision"],
                    context_recall=scores["context_recall"],
                    faithfulness=scores["faithfulness"],
                    answer_relevancy=scores["answer_relevancy"],
                    overall_score=overall_score,
                    sample_count=len(samples),
                    timestamp=datetime.utcnow(),
                    metadata={
                        "llm_model": self.llm_model,
                        "embedding_model": self.embedding_model,
                    },
                )
        
        except Exception as e:
            track_error("evaluation", e)
            logger.exception(
                "ragas_evaluation_failed",
                sample_count=len(samples),
            )
            raise QualityError(
                message="RAGAS evaluation failed",
                details={"error": str(e), "sample_count": len(samples)},
            )
    
    def _create_dataset(
        self,
        samples: List[EvaluationSample],
    ) -> Dataset:
        """Create HuggingFace dataset from samples."""
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }
        
        for sample in samples:
            data["question"].append(sample.question)
            data["answer"].append(sample.answer)
            data["contexts"].append(sample.contexts)
            data["ground_truth"].append(sample.ground_truth or "")
        
        return Dataset.from_dict(data)
    
    def _get_metrics(self, metric_names: Optional[List[str]]):
        """Get RAGAS metric objects."""
        all_metrics = {
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
        }
        
        if metric_names is None:
            return list(all_metrics.values())
        
        return [all_metrics[name] for name in metric_names]
    
    async def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate single Q&A pair.
        
        Args:
            question: User question
            answer: Generated answer
            contexts: Retrieved contexts
            ground_truth: Expected answer (optional)
        
        Returns:
            Dictionary of metric scores
        """
        sample = EvaluationSample(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )
        
        result = await self.evaluate([sample])
        
        return {
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "overall_score": result.overall_score,
        }
    
    async def batch_evaluate(
        self,
        samples: List[EvaluationSample],
        batch_size: int = 10,
    ) -> List[EvaluationResult]:
        """
        Evaluate samples in batches.
        
        Args:
            samples: All samples
            batch_size: Samples per batch
        
        Returns:
            List of EvaluationResults (one per batch)
        """
        results = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            
            logger.info(
                "evaluating_batch",
                batch_num=i // batch_size + 1,
                batch_size=len(batch),
            )
            
            result = await self.evaluate(batch)
            results.append(result)
        
        return results
    
    def check_thresholds(
        self,
        result: EvaluationResult,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> Dict[str, bool]:
        """
        Check if metrics meet thresholds.
        
        Args:
            result: Evaluation result
            thresholds: Metric thresholds (None = use defaults)
        
        Returns:
            Dictionary of pass/fail per metric
        
        Example:
            >>> checks = evaluator.check_thresholds(result, {
            ...     "faithfulness": 0.8,
            ...     "answer_relevancy": 0.7,
            ... })
            >>> if not checks["faithfulness"]:
            ...     print("Faithfulness below threshold!")
        """
        default_thresholds = {
            "context_precision": 0.7,
            "context_recall": 0.7,
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "overall_score": 0.7,
        }
        
        thresholds = thresholds or default_thresholds
        
        checks = {}
        
        for metric, threshold in thresholds.items():
            value = getattr(result, metric, None)
            if value is not None:
                checks[metric] = value >= threshold
        
        logger.info(
            "threshold_checks_completed",
            passed=sum(checks.values()),
            total=len(checks),
        )
        
        return checks


class RegressionDetector:
    """
    Detect performance regression between evaluations.
    
    Example:
        >>> detector = RegressionDetector()
        >>> is_regression = detector.detect_regression(
        ...     baseline_result,
        ...     new_result,
        ...     threshold=0.05  # 5% drop is regression
        ... )
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize regression detector.
        
        Args:
            threshold: Minimum drop to consider regression (0-1)
        """
        self.threshold = threshold
        
        logger.info("regression_detector_initialized", threshold=threshold)
    
    def detect_regression(
        self,
        baseline: EvaluationResult,
        current: EvaluationResult,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Detect regression between two evaluations.
        
        Args:
            baseline: Baseline evaluation
            current: Current evaluation
            threshold: Custom threshold (None = use default)
        
        Returns:
            Regression analysis dict
        """
        threshold = threshold or self.threshold
        
        metrics = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_relevancy",
            "overall_score",
        ]
        
        regressions = {}
        deltas = {}
        
        for metric in metrics:
            baseline_value = getattr(baseline, metric)
            current_value = getattr(current, metric)
            
            delta = current_value - baseline_value
            deltas[metric] = delta
            
            # Regression if drop > threshold
            if delta < -threshold:
                regressions[metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "delta": delta,
                    "regression": True,
                }
        
        has_regression = bool(regressions)
        
        logger.info(
            "regression_check_completed",
            has_regression=has_regression,
            regression_count=len(regressions),
        )
        
        return {
            "has_regression": has_regression,
            "regressions": regressions,
            "deltas": deltas,
            "threshold": threshold,
        }


# Convenience function

async def evaluate_rag(
    samples: List[EvaluationSample],
    llm_model: str = "gpt-4-turbo",
) -> EvaluationResult:
    """
    Convenience function for quick RAGAS evaluation.
    
    Example:
        >>> samples = [EvaluationSample(...), ...]
        >>> result = await evaluate_rag(samples)
        >>> print(result.get_summary())
    """
    evaluator = RAGASEvaluator(llm_model=llm_model)
    return await evaluator.evaluate(samples)
