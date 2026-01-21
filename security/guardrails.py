"""
Output safety checks and guardrails.

Protects against:
- Hallucinations
- PII leakage in responses
- Unsafe advice
- Low confidence answers
- Legal disclaimers missing

Production features:
- Confidence scoring
- Content filtering
- Legal compliance checks
- HITL triggers
"""

import re
from typing import Dict, List, Optional, Set, Any
from enum import Enum
from dataclasses import dataclass

from legal_assistant.core import (
    get_logger,
    get_settings,
    track_error,
    track_time,
    guardrail_check_duration,
)
from legal_assistant.core.exceptions import (
    ConfidenceError,
    QualityError,
    SecurityError,
)

logger = get_logger(__name__)
settings = get_settings()


class GuardrailStatus(str, Enum):
    """Guardrail check status."""
    
    PASS = "pass"           # All checks passed
    WARNING = "warning"     # Non-critical issues
    FAIL = "fail"           # Critical issues
    REVIEW = "review"       # Needs human review


class ConfidenceLevel(str, Enum):
    """Confidence levels."""
    
    VERY_HIGH = "very_high"  # >= 0.9
    HIGH = "high"            # >= 0.7
    MEDIUM = "medium"        # >= 0.5
    LOW = "low"              # >= 0.3
    VERY_LOW = "very_low"    # < 0.3


@dataclass
class GuardrailResult:
    """Result of guardrail checks."""
    
    status: GuardrailStatus
    confidence: float
    confidence_level: ConfidenceLevel
    issues: List[str]
    warnings: List[str]
    needs_review: bool
    safe_for_output: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "issues": self.issues,
            "warnings": self.warnings,
            "needs_review": self.needs_review,
            "safe_for_output": self.safe_for_output,
            "metadata": self.metadata,
        }


class OutputGuardrails:
    """
    Output safety checks and guardrails.
    
    Features:
    - Confidence scoring
    - PII leakage detection
    - Legal disclaimer validation
    - Hallucination detection
    - HITL triggering
    
    Example:
        >>> guardrails = OutputGuardrails()
        >>> result = guardrails.check_output(
        ...     answer="Договор купли-продажи...",
        ...     sources=[{...}],
        ...     confidence=0.85
        ... )
        >>> if result.safe_for_output:
        ...     return answer
        >>> elif result.needs_review:
        ...     queue_for_review(answer)
    """
    
    # Patterns for unsafe content
    UNSAFE_PATTERNS = [
        r"я не (знаю|уверен|могу сказать)",
        r"(возможно|вероятно|может быть),?\s+но я не уверен",
        r"информация может быть (устаревшей|неточной)",
        r"я (придумал|выдумал|сочинил)",
    ]
    
    # Legal disclaimer patterns
    LEGAL_DISCLAIMER_PATTERNS = [
        r"не является юридической консультацией",
        r"проконсультируйтесь с (юристом|адвокатом)",
        r"для получения юридической консультации",
        r"это общая информация",
    ]
    
    # Hallucination indicators
    HALLUCINATION_PATTERNS = [
        r"в источнике №\d+ не упоминается",
        r"источник не содержит",
        r"я не нашел (подтверждения|информации)",
        r"противоречит источнику",
    ]
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        require_disclaimer: bool = True,
        check_pii_leakage: bool = True,
    ):
        """
        Initialize guardrails.
        
        Args:
            confidence_threshold: Minimum confidence for auto-approval
            require_disclaimer: Require legal disclaimer
            check_pii_leakage: Check for PII in output
        """
        self.confidence_threshold = confidence_threshold
        self.require_disclaimer = require_disclaimer
        self.check_pii_leakage = check_pii_leakage
        
        logger.info(
            "guardrails_initialized",
            confidence_threshold=confidence_threshold,
            require_disclaimer=require_disclaimer,
        )
    
    def check_output(
        self,
        answer: str,
        sources: List[Dict],
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ) -> GuardrailResult:
        """
        Run all guardrail checks on output.
        
        Args:
            answer: Generated answer
            sources: Retrieved source documents
            confidence: Confidence score (0-1)
            metadata: Additional metadata
        
        Returns:
            GuardrailResult with check results
        
        Example:
            >>> result = guardrails.check_output(
            ...     answer="Договор купли-продажи — это...",
            ...     sources=[{"text": "...", "score": 0.85}],
            ...     confidence=0.82
            ... )
            >>> print(result.status)
            GuardrailStatus.PASS
        """
        issues = []
        warnings = []
        check_metadata = metadata or {}
        
        try:
            with track_time(
                guardrail_check_duration,
                {"check_type": "full"}
            ):
                # 1. Confidence check
                confidence_level = self._check_confidence(
                    confidence, issues, warnings
                )
                
                # 2. Answer quality checks
                self._check_answer_quality(answer, issues, warnings)
                
                # 3. Source attribution
                self._check_source_attribution(
                    answer, sources, issues, warnings
                )
                
                # 4. Legal disclaimer
                if self.require_disclaimer:
                    self._check_legal_disclaimer(answer, issues, warnings)
                
                # 5. PII leakage
                if self.check_pii_leakage:
                    self._check_pii_leakage(answer, issues, warnings)
                
                # 6. Hallucination detection
                self._check_hallucination(answer, issues, warnings)
                
                # 7. Unsafe content
                self._check_unsafe_content(answer, issues, warnings)
                
                # Determine status
                if issues:
                    status = GuardrailStatus.FAIL
                    safe_for_output = False
                elif warnings:
                    status = GuardrailStatus.WARNING
                    safe_for_output = True
                else:
                    status = GuardrailStatus.PASS
                    safe_for_output = True
                
                # Determine if needs review
                needs_review = (
                    confidence is not None
                    and confidence < self.confidence_threshold
                ) or bool(issues)
                
                logger.info(
                    "guardrail_check_completed",
                    status=status.value,
                    confidence=confidence,
                    issue_count=len(issues),
                    warning_count=len(warnings),
                    needs_review=needs_review,
                )
                
                return GuardrailResult(
                    status=status,
                    confidence=confidence or 0.0,
                    confidence_level=confidence_level,
                    issues=issues,
                    warnings=warnings,
                    needs_review=needs_review,
                    safe_for_output=safe_for_output,
                    metadata=check_metadata,
                )
        
        except Exception as e:
            track_error("security", e)
            logger.exception("guardrail_check_failed")
            
            # Fail-safe: mark as needs review
            return GuardrailResult(
                status=GuardrailStatus.FAIL,
                confidence=0.0,
                confidence_level=ConfidenceLevel.VERY_LOW,
                issues=[f"Guardrail check failed: {str(e)}"],
                warnings=[],
                needs_review=True,
                safe_for_output=False,
                metadata={"error": str(e)},
            )
    
    def _check_confidence(
        self,
        confidence: Optional[float],
        issues: List[str],
        warnings: List[str],
    ) -> ConfidenceLevel:
        """Check confidence score."""
        if confidence is None:
            warnings.append("No confidence score provided")
            return ConfidenceLevel.MEDIUM
        
        # Validate range
        if not 0 <= confidence <= 1:
            issues.append(f"Invalid confidence score: {confidence}")
            return ConfidenceLevel.VERY_LOW
        
        # Categorize
        if confidence >= 0.9:
            level = ConfidenceLevel.VERY_HIGH
        elif confidence >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        elif confidence >= 0.3:
            level = ConfidenceLevel.LOW
        else:
            level = ConfidenceLevel.VERY_LOW
        
        # Check threshold
        if confidence < self.confidence_threshold:
            warnings.append(
                f"Confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
        
        if confidence < 0.5:
            issues.append(
                f"Confidence too low ({confidence:.2f}), "
                "answer may be unreliable"
            )
        
        return level
    
    def _check_answer_quality(
        self,
        answer: str,
        issues: List[str],
        warnings: List[str],
    ):
        """Check answer quality."""
        # Check empty
        if not answer or not answer.strip():
            issues.append("Answer is empty")
            return
        
        # Check minimum length
        if len(answer) < 10:
            issues.append("Answer too short (< 10 chars)")
        
        # Check maximum length
        if len(answer) > 5000:
            warnings.append("Answer very long (> 5000 chars)")
        
        # Check for placeholder text
        placeholders = ["TODO", "TBD", "...", "xxx", "[placeholder]"]
        for placeholder in placeholders:
            if placeholder.lower() in answer.lower():
                warnings.append(f"Answer contains placeholder: {placeholder}")
        
        # Check sentence structure
        sentences = answer.split(".")
        if len(sentences) == 1 and len(answer) > 200:
            warnings.append("Answer is one long sentence")
    
    def _check_source_attribution(
        self,
        answer: str,
        sources: List[Dict],
        issues: List[str],
        warnings: List[str],
    ):
        """Check source attribution and citations."""
        # Check if sources provided
        if not sources:
            warnings.append("No sources provided")
            return
        
        # Check for citation markers
        citation_patterns = [
            r"\[источник\s*\d+\]",
            r"\[Источник:\s*[^\]]+\]",
            r"согласно источнику",
            r"в документе",
        ]
        
        has_citations = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in citation_patterns
        )
        
        if not has_citations and len(sources) > 0:
            warnings.append("Answer lacks explicit source citations")
        
        # Check source relevance scores
        low_score_sources = [
            s for s in sources
            if s.get("score", 1.0) < 0.5
        ]
        
        if low_score_sources:
            warnings.append(
                f"{len(low_score_sources)} sources have low relevance scores"
            )
    
    def _check_legal_disclaimer(
        self,
        answer: str,
        issues: List[str],
        warnings: List[str],
    ):
        """Check for legal disclaimer."""
        has_disclaimer = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in self.LEGAL_DISCLAIMER_PATTERNS
        )
        
        if not has_disclaimer:
            warnings.append(
                "Answer lacks legal disclaimer. "
                "Consider adding: 'Это общая информация, "
                "не является юридической консультацией'"
            )
    
    def _check_pii_leakage(
        self,
        answer: str,
        issues: List[str],
        warnings: List[str],
    ):
        """Check for PII leakage in output."""
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, answer):
            issues.append("Answer contains email address")
        
        # Phone pattern (Russian)
        phone_pattern = r'\+?7[\s-]?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}'
        if re.search(phone_pattern, answer):
            issues.append("Answer contains phone number")
        
        # Passport pattern
        passport_pattern = r'\b\d{4}\s?\d{6}\b'
        if re.search(passport_pattern, answer):
            issues.append("Answer may contain passport number")
        
        # Credit card pattern
        card_pattern = r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b'
        if re.search(card_pattern, answer):
            issues.append("Answer may contain credit card number")
    
    def _check_hallucination(
        self,
        answer: str,
        issues: List[str],
        warnings: List[str],
    ):
        """Check for hallucination indicators."""
        for pattern in self.HALLUCINATION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                warnings.append(
                    "Answer contains possible hallucination indicator"
                )
                break
    
    def _check_unsafe_content(
        self,
        answer: str,
        issues: List[str],
        warnings: List[str],
    ):
        """Check for unsafe content patterns."""
        for pattern in self.UNSAFE_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                warnings.append(
                    "Answer contains uncertainty/lack of knowledge indicator"
                )
                break
    
    def should_trigger_hitl(
        self,
        result: GuardrailResult,
        task_type: str = "general",
    ) -> bool:
        """
        Determine if Human-in-the-Loop should be triggered.
        
        Args:
            result: Guardrail check result
            task_type: Type of task (general, legal_opinion, contract_review)
        
        Returns:
            True if HITL should be triggered
        
        Example:
            >>> if guardrails.should_trigger_hitl(result, "contract_review"):
            ...     queue_for_human_review(answer)
        """
        # Always trigger for critical issues
        if result.status == GuardrailStatus.FAIL:
            logger.info("hitl_triggered", reason="critical_issues")
            return True
        
        # Trigger for low confidence
        if result.confidence < self.confidence_threshold:
            logger.info(
                "hitl_triggered",
                reason="low_confidence",
                confidence=result.confidence,
            )
            return True
        
        # Task-specific triggers
        if task_type == "legal_opinion":
            # More conservative for legal opinions
            if result.confidence < 0.85:
                logger.info("hitl_triggered", reason="legal_opinion_caution")
                return True
        
        elif task_type == "contract_review":
            # Always review contract modifications
            logger.info("hitl_triggered", reason="contract_modification")
            return True
        
        elif task_type == "risk_assessment":
            # Conservative for risk assessments
            if result.confidence < 0.8 or result.warnings:
                logger.info("hitl_triggered", reason="risk_assessment_caution")
                return True
        
        return False
    
    def add_disclaimer(
        self,
        answer: str,
        disclaimer_type: str = "general",
    ) -> str:
        """
        Add legal disclaimer to answer.
        
        Args:
            answer: Original answer
            disclaimer_type: Type of disclaimer (general, legal, medical)
        
        Returns:
            Answer with disclaimer
        """
        disclaimers = {
            "general": (
                "\n\n---\n"
                "⚠️ Это общая информация, не является юридической консультацией. "
                "Для получения профессиональной помощи обратитесь к квалифицированному юристу."
            ),
            "legal": (
                "\n\n---\n"
                "⚠️ ВАЖНО: Данная информация носит справочный характер и не может "
                "рассматриваться как юридическая консультация. Законодательство может "
                "измениться. Для решения конкретной правовой ситуации обратитесь к адвокату."
            ),
            "contract": (
                "\n\n---\n"
                "⚠️ Анализ носит предварительный характер. Перед подписанием договора "
                "рекомендуется консультация с юристом."
            ),
        }
        
        disclaimer = disclaimers.get(disclaimer_type, disclaimers["general"])
        return answer + disclaimer


class ModelOutputLimits:
    """
    Model output limits and cost controls.
    
    Prevents runaway costs and token usage.
    """
    
    def __init__(
        self,
        max_tokens_per_request: int = 4000,
        max_cost_per_request: float = 1.0,
        daily_token_budget: Optional[int] = None,
        daily_cost_budget: Optional[float] = None,
    ):
        """
        Initialize output limits.
        
        Args:
            max_tokens_per_request: Max tokens per single request
            max_cost_per_request: Max cost ($) per request
            daily_token_budget: Daily token budget (None = unlimited)
            daily_cost_budget: Daily cost budget ($) (None = unlimited)
        """
        self.max_tokens_per_request = max_tokens_per_request
        self.max_cost_per_request = max_cost_per_request
        self.daily_token_budget = daily_token_budget
        self.daily_cost_budget = daily_cost_budget
        
        # Runtime tracking (reset daily)
        self.tokens_used_today = 0
        self.cost_used_today = 0.0
        
        logger.info(
            "model_limits_initialized",
            max_tokens=max_tokens_per_request,
            max_cost=max_cost_per_request,
        )
    
    def check_limits(
        self,
        estimated_tokens: int,
        estimated_cost: float,
    ) -> bool:
        """
        Check if request is within limits.
        
        Args:
            estimated_tokens: Estimated token usage
            estimated_cost: Estimated cost ($)
        
        Returns:
            True if within limits
        
        Raises:
            QualityError: If limits exceeded
        """
        # Per-request limits
        if estimated_tokens > self.max_tokens_per_request:
            raise QualityError(
                message="Request exceeds token limit",
                details={
                    "estimated_tokens": estimated_tokens,
                    "max_tokens": self.max_tokens_per_request,
                },
            )
        
        if estimated_cost > self.max_cost_per_request:
            raise QualityError(
                message="Request exceeds cost limit",
                details={
                    "estimated_cost": estimated_cost,
                    "max_cost": self.max_cost_per_request,
                },
            )
        
        # Daily budgets
        if self.daily_token_budget:
            if self.tokens_used_today + estimated_tokens > self.daily_token_budget:
                raise QualityError(
                    message="Daily token budget exceeded",
                    details={
                        "tokens_used": self.tokens_used_today,
                        "budget": self.daily_token_budget,
                    },
                )
        
        if self.daily_cost_budget:
            if self.cost_used_today + estimated_cost > self.daily_cost_budget:
                raise QualityError(
                    message="Daily cost budget exceeded",
                    details={
                        "cost_used": self.cost_used_today,
                        "budget": self.daily_cost_budget,
                    },
                )
        
        return True
    
    def track_usage(self, tokens: int, cost: float):
        """Track token and cost usage."""
        self.tokens_used_today += tokens
        self.cost_used_today += cost
        
        logger.debug(
            "usage_tracked",
            tokens=tokens,
            cost=cost,
            total_tokens=self.tokens_used_today,
            total_cost=self.cost_used_today,
        )
    
    def reset_daily_usage(self):
        """Reset daily usage counters."""
        logger.info(
            "daily_usage_reset",
            tokens_used=self.tokens_used_today,
            cost_used=self.cost_used_today,
        )
        
        self.tokens_used_today = 0
        self.cost_used_today = 0.0


# Convenience functions

def check_output(
    answer: str,
    sources: List[Dict],
    confidence: Optional[float] = None,
    **kwargs,
) -> GuardrailResult:
    """Convenience function for output checking."""
    guardrails = OutputGuardrails(**kwargs)
    return guardrails.check_output(answer, sources, confidence)


def add_disclaimer(answer: str, disclaimer_type: str = "general") -> str:
    """Convenience function for adding disclaimer."""
    guardrails = OutputGuardrails()
    return guardrails.add_disclaimer(answer, disclaimer_type)
