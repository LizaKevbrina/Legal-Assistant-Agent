"""
LLM-as-a-Judge evaluation.

Uses LLM to evaluate:
- Answer quality
- Correctness
- Completeness
- Tone appropriateness
- Legal accuracy

Production features:
- Multiple judge models
- Structured scoring
- Explanation generation
- Consensus voting (multi-judge)
- Cost tracking
"""

from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_time,
    track_error,
    track_llm_tokens,
    evaluation_duration,
)
from legal_assistant.core.exceptions import QualityError
from legal_assistant.generation import LLMRouter

logger = get_logger(__name__)
settings = get_settings()


class JudgmentCriterion(str, Enum):
    """Evaluation criteria."""
    
    CORRECTNESS = "correctness"           # Factually correct
    COMPLETENESS = "completeness"         # Covers all aspects
    RELEVANCY = "relevancy"               # Relevant to question
    CLARITY = "clarity"                   # Clear and understandable
    TONE = "tone"                         # Appropriate tone
    LEGAL_ACCURACY = "legal_accuracy"     # Legal terminology correct
    CITATIONS = "citations"               # Proper source attribution


@dataclass
class JudgmentScore:
    """Single criterion judgment."""
    
    criterion: JudgmentCriterion
    score: float  # 0-1
    explanation: str
    examples: Optional[List[str]] = None
    
    def to_dict(self) -> Dict:
        return {
            "criterion": self.criterion.value,
            "score": self.score,
            "explanation": self.explanation,
            "examples": self.examples or [],
        }


@dataclass
class LLMJudgment:
    """Complete LLM judgment."""
    
    overall_score: float  # 0-1
    scores: List[JudgmentScore]
    verdict: Literal["excellent", "good", "acceptable", "poor"]
    summary: str
    recommendations: List[str]
    judge_model: str
    timestamp: datetime
    cost_usd: float
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "scores": [s.to_dict() for s in self.scores],
            "verdict": self.verdict,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "judge_model": self.judge_model,
            "timestamp": self.timestamp.isoformat(),
            "cost_usd": self.cost_usd,
        }
    
    def get_summary(self) -> str:
        """Human-readable summary."""
        return (
            f"LLM Judge Verdict: {self.verdict.upper()} ({self.overall_score:.2f})\n"
            f"Judge Model: {self.judge_model}\n"
            f"\nScores:\n" +
            "\n".join(
                f"  - {s.criterion.value}: {s.score:.2f} - {s.explanation[:50]}..."
                for s in self.scores
            ) +
            f"\n\nSummary: {self.summary}\n"
            f"\nRecommendations:\n" +
            "\n".join(f"  - {r}" for r in self.recommendations)
        )


class LLMJudge:
    """
    LLM-as-a-Judge evaluator.
    
    Features:
    - Multi-criterion evaluation
    - Structured scoring
    - Explanation generation
    - Multiple judge models
    - Consensus voting
    
    Example:
        >>> judge = LLMJudge(model="gpt-4-turbo")
        >>> judgment = await judge.evaluate(
        ...     question="Что такое договор?",
        ...     answer="Договор — это соглашение...",
        ...     reference="Договор определяется...",
        ...     criteria=[
        ...         JudgmentCriterion.CORRECTNESS,
        ...         JudgmentCriterion.COMPLETENESS,
        ...     ]
        ... )
        >>> print(judgment.get_summary())
    """
    
    JUDGE_SYSTEM_PROMPT = """Ты — эксперт по оценке качества ответов юридического AI-ассистента.

Твоя задача: оценить качество ответа по заданным критериям.

Для каждого критерия:
1. Дай оценку от 0 до 1 (0 = очень плохо, 1 = отлично)
2. Объясни оценку (2-3 предложения)
3. Приведи конкретные примеры из ответа (если применимо)

Будь объективным, справедливым и конструктивным.
"""
    
    CRITERIA_DESCRIPTIONS = {
        JudgmentCriterion.CORRECTNESS: "Насколько ответ фактически верен?",
        JudgmentCriterion.COMPLETENESS: "Полностью ли раскрыт вопрос?",
        JudgmentCriterion.RELEVANCY: "Насколько ответ релевантен вопросу?",
        JudgmentCriterion.CLARITY: "Насколько ответ понятен и структурирован?",
        JudgmentCriterion.TONE: "Подходит ли тон ответа для юридического контекста?",
        JudgmentCriterion.LEGAL_ACCURACY: "Корректна ли юридическая терминология?",
        JudgmentCriterion.CITATIONS: "Правильно ли указаны источники?",
    }
    
    def __init__(
        self,
        model: str = "gpt-4-turbo",
        temperature: float = 0.3,
    ):
        """
        Initialize LLM judge.
        
        Args:
            model: Judge model name
            temperature: Temperature for judgment (lower = more consistent)
        """
        self.model = model
        self.temperature = temperature
        self.llm_router = LLMRouter()
        
        logger.info(
            "llm_judge_initialized",
            model=model,
            temperature=temperature,
        )
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        criteria: Optional[List[JudgmentCriterion]] = None,
    ) -> LLMJudgment:
        """
        Evaluate answer quality using LLM.
        
        Args:
            question: User question
            answer: Generated answer to evaluate
            reference: Ground truth answer (optional)
            contexts: Retrieved contexts (optional)
            criteria: Criteria to evaluate (None = all)
        
        Returns:
            LLMJudgment with scores and recommendations
        
        Example:
            >>> judgment = await judge.evaluate(
            ...     question="Что такое договор?",
            ...     answer="Договор — соглашение сторон...",
            ...     reference="Договор — это взаимное...",
            ...     criteria=[JudgmentCriterion.CORRECTNESS]
            ... )
        """
        try:
            with track_time(
                evaluation_duration,
                {"metric": "llm_judge", "model": self.model}
            ):
                # Default criteria
                if criteria is None:
                    criteria = [
                        JudgmentCriterion.CORRECTNESS,
                        JudgmentCriterion.COMPLETENESS,
                        JudgmentCriterion.RELEVANCY,
                        JudgmentCriterion.CLARITY,
                    ]
                
                # Build evaluation prompt
                prompt = self._build_evaluation_prompt(
                    question, answer, reference, contexts, criteria
                )
                
                logger.debug(
                    "llm_judge_evaluation_started",
                    criteria_count=len(criteria),
                )
                
                # Call LLM
                async with self.llm_router as router:
                    response = await router.generate(
                        messages=[
                            {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        model=self.model,
                        temperature=self.temperature,
                        max_tokens=2000,
                    )
                
                # Track tokens
                track_llm_tokens(
                    provider=response.provider,
                    model=response.model,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    cost_usd=response.cost_usd,
                )
                
                # Parse response
                judgment = self._parse_judgment(
                    response.content,
                    criteria,
                    response.model,
                    response.cost_usd,
                )
                
                logger.info(
                    "llm_judge_evaluation_completed",
                    overall_score=judgment.overall_score,
                    verdict=judgment.verdict,
                    cost_usd=judgment.cost_usd,
                )
                
                return judgment
        
        except Exception as e:
            track_error("evaluation", e)
            logger.exception("llm_judge_evaluation_failed")
            raise QualityError(
                message="LLM judge evaluation failed",
                details={"error": str(e)},
            )
    
    def _build_evaluation_prompt(
        self,
        question: str,
        answer: str,
        reference: Optional[str],
        contexts: Optional[List[str]],
        criteria: List[JudgmentCriterion],
    ) -> str:
        """Build evaluation prompt."""
        sections = [
            "# ЗАДАЧА ОЦЕНКИ",
            "",
            "## Вопрос пользователя:",
            question,
            "",
            "## Ответ системы (для оценки):",
            answer,
        ]
        
        if reference:
            sections.extend([
                "",
                "## Эталонный ответ (для сравнения):",
                reference,
            ])
        
        if contexts:
            sections.extend([
                "",
                "## Контексты (источники):",
            ])
            for i, ctx in enumerate(contexts[:3], 1):
                sections.append(f"\n{i}. {ctx[:200]}...")
        
        sections.extend([
            "",
            "## Критерии оценки:",
            "",
        ])
        
        for criterion in criteria:
            desc = self.CRITERIA_DESCRIPTIONS.get(criterion, "")
            sections.append(f"- **{criterion.value}**: {desc}")
        
        sections.extend([
            "",
            "## ФОРМАТ ОТВЕТА:",
            "",
            "Для КАЖДОГО критерия предоставь:",
            "",
            "### [Название критерия]",
            "- **Оценка**: [число от 0 до 1]",
            "- **Объяснение**: [2-3 предложения]",
            "- **Примеры**: [конкретные фрагменты из ответа, если применимо]",
            "",
            "Затем:",
            "",
            "### ОБЩИЙ ВЕРДИКТ",
            "- **Итоговая оценка**: [среднее от 0 до 1]",
            "- **Категория**: [excellent/good/acceptable/poor]",
            "- **Краткое резюме**: [1-2 предложения]",
            "- **Рекомендации**: [список улучшений, если есть]",
        ])
        
        return "\n".join(sections)
    
    def _parse_judgment(
        self,
        response_text: str,
        criteria: List[JudgmentCriterion],
        model: str,
        cost: float,
    ) -> LLMJudgment:
        """Parse LLM judgment response."""
        # Simple regex-based parsing
        # In production, use more robust parsing or structured output
        
        import re
        
        scores = []
        
        # Parse individual criteria scores
        for criterion in criteria:
            pattern = rf"###?\s*{criterion.value}.*?Оценка.*?(\d+\.?\d*)"
            match = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
            
            if match:
                score_value = float(match.group(1))
                if score_value > 1:  # Handle 0-100 scale
                    score_value /= 100
                
                # Extract explanation
                expl_pattern = rf"Объяснение.*?:(.*?)(?=Примеры|###|$)"
                expl_match = re.search(
                    expl_pattern,
                    response_text[match.start():],
                    re.DOTALL,
                )
                explanation = (
                    expl_match.group(1).strip()
                    if expl_match
                    else "Нет объяснения"
                )
                
                scores.append(
                    JudgmentScore(
                        criterion=criterion,
                        score=score_value,
                        explanation=explanation[:200],
                    )
                )
        
        # Calculate overall score
        if scores:
            overall_score = sum(s.score for s in scores) / len(scores)
        else:
            overall_score = 0.5  # Fallback
        
        # Determine verdict
        if overall_score >= 0.9:
            verdict = "excellent"
        elif overall_score >= 0.7:
            verdict = "good"
        elif overall_score >= 0.5:
            verdict = "acceptable"
        else:
            verdict = "poor"
        
        # Extract summary
        summary_pattern = r"Краткое резюме.*?:(.*?)(?=Рекомендации|$)"
        summary_match = re.search(
            summary_pattern,
            response_text,
            re.DOTALL | re.IGNORECASE,
        )
        summary = (
            summary_match.group(1).strip()[:500]
            if summary_match
            else "Нет резюме"
        )
        
        # Extract recommendations
        rec_pattern = r"Рекомендации.*?:(.*?)$"
        rec_match = re.search(rec_pattern, response_text, re.DOTALL | re.IGNORECASE)
        
        recommendations = []
        if rec_match:
            rec_text = rec_match.group(1)
            recommendations = [
                line.strip("- •*")
                for line in rec_text.split("\n")
                if line.strip() and line.strip().startswith(("- ", "• ", "* "))
            ][:5]
        
        return LLMJudgment(
            overall_score=overall_score,
            scores=scores,
            verdict=verdict,
            summary=summary,
            recommendations=recommendations,
            judge_model=model,
            timestamp=datetime.utcnow(),
            cost_usd=cost,
        )
    
    async def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        criteria: Optional[List[JudgmentCriterion]] = None,
    ) -> List[LLMJudgment]:
        """
        Evaluate multiple samples.
        
        Args:
            samples: List of dicts with 'question', 'answer', etc.
            criteria: Criteria to evaluate
        
        Returns:
            List of LLMJudgments
        """
        tasks = [
            self.evaluate(
                question=s["question"],
                answer=s["answer"],
                reference=s.get("reference"),
                contexts=s.get("contexts"),
                criteria=criteria,
            )
            for s in samples
        ]
        
        logger.info("evaluating_batch", sample_count=len(samples))
        
        judgments = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_judgments = [
            j for j in judgments
            if isinstance(j, LLMJudgment)
        ]
        
        logger.info(
            "batch_evaluation_completed",
            total=len(samples),
            successful=len(valid_judgments),
        )
        
        return valid_judgments


class ConsensusJudge:
    """
    Multi-judge consensus evaluation.
    
    Uses multiple LLM judges and aggregates their verdicts.
    
    Example:
        >>> consensus = ConsensusJudge(
        ...     judges=["gpt-4-turbo", "claude-3-sonnet"]
        ... )
        >>> result = await consensus.evaluate(question, answer)
    """
    
    def __init__(
        self,
        judge_models: List[str],
        aggregation: Literal["mean", "median", "max", "min"] = "mean",
    ):
        """
        Initialize consensus judge.
        
        Args:
            judge_models: List of judge model names
            aggregation: Score aggregation method
        """
        self.judges = [LLMJudge(model=m) for m in judge_models]
        self.aggregation = aggregation
        
        logger.info(
            "consensus_judge_initialized",
            judge_count=len(self.judges),
            aggregation=aggregation,
        )
    
    async def evaluate(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        contexts: Optional[List[str]] = None,
        criteria: Optional[List[JudgmentCriterion]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate using multiple judges and aggregate.
        
        Returns:
            Dict with aggregated scores and individual judgments
        """
        # Evaluate with all judges in parallel
        tasks = [
            judge.evaluate(question, answer, reference, contexts, criteria)
            for judge in self.judges
        ]
        
        logger.info("consensus_evaluation_started", judge_count=len(self.judges))
        
        judgments = await asyncio.gather(*tasks)
        
        # Aggregate scores
        aggregated = self._aggregate_judgments(judgments)
        
        logger.info(
            "consensus_evaluation_completed",
            overall_score=aggregated["overall_score"],
            agreement=aggregated["agreement"],
        )
        
        return {
            "aggregated": aggregated,
            "individual_judgments": [j.to_dict() for j in judgments],
        }
    
    def _aggregate_judgments(
        self,
        judgments: List[LLMJudgment],
    ) -> Dict[str, Any]:
        """Aggregate multiple judgments."""
        import statistics
        
        overall_scores = [j.overall_score for j in judgments]
        
        # Aggregate overall score
        if self.aggregation == "mean":
            agg_score = statistics.mean(overall_scores)
        elif self.aggregation == "median":
            agg_score = statistics.median(overall_scores)
        elif self.aggregation == "max":
            agg_score = max(overall_scores)
        else:  # min
            agg_score = min(overall_scores)
        
        # Calculate agreement (std dev)
        agreement = 1 - statistics.stdev(overall_scores) if len(overall_scores) > 1 else 1.0
        
        # Aggregate criterion scores
        criterion_scores = {}
        for judgment in judgments:
            for score in judgment.scores:
                if score.criterion not in criterion_scores:
                    criterion_scores[score.criterion] = []
                criterion_scores[score.criterion].append(score.score)
        
        aggregated_criteria = {
            criterion: statistics.mean(scores)
            for criterion, scores in criterion_scores.items()
        }
        
        # Collect all recommendations
        all_recommendations = []
        for judgment in judgments:
            all_recommendations.extend(judgment.recommendations)
        
        # Deduplicate recommendations
        unique_recommendations = list(set(all_recommendations))[:5]
        
        return {
            "overall_score": agg_score,
            "agreement": agreement,
            "criterion_scores": aggregated_criteria,
            "recommendations": unique_recommendations,
            "verdict_distribution": {
                j.verdict: sum(1 for jj in judgments if jj.verdict == j.verdict)
                for j in judgments
            },
        }


# Convenience function

async def judge_answer(
    question: str,
    answer: str,
    reference: Optional[str] = None,
    model: str = "gpt-4-turbo",
) -> LLMJudgment:
    """
    Convenience function for quick LLM judgment.
    
    Example:
        >>> judgment = await judge_answer(
        ...     question="Что такое договор?",
        ...     answer="Договор — это соглашение...",
        ...     reference="Договор определяется как...",
        ... )
        >>> print(judgment.verdict)
    """
    judge = LLMJudge(model=model)
    return await judge.evaluate(question, answer, reference)
