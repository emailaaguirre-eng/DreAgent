"""
=============================================================================
HUMMINGBIRD-LEA - Confidence Scoring System
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Multi-factor confidence scoring for agent responses.

Factors considered:
1. Response language analysis (hedging words, certainty markers)
2. Question complexity assessment
3. Knowledge domain matching
4. Source availability
5. Reasoning chain completeness

Design Philosophy:
- If confidence < 85%, the agent should express uncertainty
- Lower confidence = more likely to ask clarifying questions
- Critical actions require higher confidence thresholds
=============================================================================
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ConfidenceLevel(Enum):
    """Agent's confidence in its response"""
    VERY_LOW = "very_low"    # < 40% - Should not proceed
    LOW = "low"              # 40-60% - Express significant uncertainty
    MEDIUM = "medium"        # 60-85% - Express some uncertainty
    HIGH = "high"            # 85-95% - Confident but acknowledge limits
    VERY_HIGH = "very_high"  # > 95% - Highly confident


class ConfidenceFactor(Enum):
    """Factors that affect confidence"""
    UNCERTAINTY_LANGUAGE = "uncertainty_language"
    CERTAINTY_LANGUAGE = "certainty_language"
    QUESTION_COMPLEXITY = "question_complexity"
    KNOWLEDGE_DOMAIN = "knowledge_domain"
    HEDGING_PHRASES = "hedging_phrases"
    SPECIFICITY = "specificity"
    SOURCE_AVAILABILITY = "source_availability"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ConfidenceFactorScore:
    """Score for a single confidence factor"""
    factor: ConfidenceFactor
    score: float  # 0.0 to 1.0
    weight: float  # How much this factor matters
    explanation: str
    evidence: Optional[List[str]] = None  # What triggered this score


@dataclass
class ConfidenceAssessment:
    """Complete confidence assessment for a response"""
    raw_score: float              # 0.0 to 1.0
    level: ConfidenceLevel
    factors: List[ConfidenceFactorScore]
    should_express_uncertainty: bool
    should_ask_clarification: bool
    uncertainty_phrases: List[str]  # Suggested phrases to use
    explanation: str               # Human-readable summary

    @property
    def percentage(self) -> int:
        """Get confidence as a percentage"""
        return int(self.raw_score * 100)


# =============================================================================
# Confidence Scorer
# =============================================================================

class ConfidenceScorer:
    """
    Multi-factor confidence scoring system.

    Analyzes both the question and response to determine
    how confident the agent should be in its answer.
    """

    # -------------------------------------------------------------------------
    # Uncertainty Markers (indicate lower confidence)
    # -------------------------------------------------------------------------
    UNCERTAINTY_MARKERS = [
        # Strong uncertainty
        (r"\bi(?:'m| am) not sure\b", 0.3, "strong"),
        (r"\bi don(?:'t| not) know\b", 0.2, "strong"),
        (r"\bi(?:'m| am) uncertain\b", 0.3, "strong"),
        (r"\bi can(?:'t| not) (?:say|tell|determine)\b", 0.3, "strong"),

        # Moderate uncertainty
        (r"\b(?:might|may|could) be\b", 0.5, "moderate"),
        (r"\bpossibly\b", 0.5, "moderate"),
        (r"\bperhaps\b", 0.5, "moderate"),
        (r"\bprobably\b", 0.6, "moderate"),

        # Mild uncertainty
        (r"\bi think\b", 0.7, "mild"),
        (r"\bi believe\b", 0.7, "mild"),
        (r"\bit seems\b", 0.7, "mild"),
        (r"\bapparently\b", 0.7, "mild"),
        (r"\bas far as i know\b", 0.6, "mild"),
        (r"\bif i recall\b", 0.6, "mild"),
    ]

    # -------------------------------------------------------------------------
    # Certainty Markers (indicate higher confidence)
    # -------------------------------------------------------------------------
    CERTAINTY_MARKERS = [
        (r"\bdefinitely\b", 0.95),
        (r"\bcertainly\b", 0.9),
        (r"\babsolutely\b", 0.95),
        (r"\bwithout (?:a )?doubt\b", 0.95),
        (r"\bclearly\b", 0.85),
        (r"\bobviously\b", 0.85),
        (r"\bof course\b", 0.85),
        (r"\bi(?:'m| am) confident\b", 0.9),
        (r"\bi(?:'m| am) certain\b", 0.95),
    ]

    # -------------------------------------------------------------------------
    # Hedging Phrases (soften claims, indicate uncertainty)
    # -------------------------------------------------------------------------
    HEDGING_PHRASES = [
        (r"\bgenerally speaking\b", 0.7),
        (r"\bin most cases\b", 0.7),
        (r"\btypically\b", 0.75),
        (r"\busually\b", 0.75),
        (r"\boften\b", 0.7),
        (r"\bsometimes\b", 0.6),
        (r"\bit depends\b", 0.5),
        (r"\bto some extent\b", 0.6),
        (r"\bin theory\b", 0.6),
        (r"\bideally\b", 0.7),
    ]

    # -------------------------------------------------------------------------
    # Question Complexity Markers
    # -------------------------------------------------------------------------
    COMPLEX_QUESTION_MARKERS = [
        r"\bwhy\b",           # Causation questions are harder
        r"\bhow does\b",      # Mechanism questions
        r"\bexplain\b",       # Explanation requests
        r"\bcompare\b",       # Comparison questions
        r"\banalyze\b",       # Analysis requests
        r"\bpredict\b",       # Prediction requests
        r"\bwhat if\b",       # Hypothetical scenarios
        r"\bshould\b",        # Normative questions
        r"\bbest way\b",      # Optimization questions
    ]

    SIMPLE_QUESTION_MARKERS = [
        r"\bwhat is\b",       # Definition questions
        r"\bwho is\b",        # Identity questions
        r"\bwhere is\b",      # Location questions
        r"\bwhen\b",          # Time questions
        r"\blist\b",          # Enumeration requests
        r"\bdefine\b",        # Definition requests
    ]

    # -------------------------------------------------------------------------
    # Knowledge Domain Confidence (how well we know these topics)
    # -------------------------------------------------------------------------
    HIGH_CONFIDENCE_DOMAINS = [
        # Agent capabilities
        r"\b(schedule|calendar|meeting|appointment)\b",
        r"\b(email|message|send|draft)\b",
        r"\b(task|todo|reminder)\b",
        r"\b(document|file|folder)\b",
        # Basic operations
        r"\b(help|explain|describe)\b",
        r"\b(create|make|generate)\b",
    ]

    LOW_CONFIDENCE_DOMAINS = [
        # Factual claims that could be outdated
        r"\bcurrent (?:price|rate|value)\b",
        r"\blatest news\b",
        r"\bright now\b",
        r"\btoday's\b",
        # Personal/private information
        r"\byour (?:password|account)\b",
        # Predictions
        r"\bwill (?:happen|be)\b",
        r"\bfuture\b",
    ]

    def __init__(
        self,
        base_confidence: float = 0.85,
        threshold: float = 0.85,
    ):
        """
        Initialize the scorer.

        Args:
            base_confidence: Starting confidence before adjustments
            threshold: Below this, agent should express uncertainty
        """
        self.base_confidence = base_confidence
        self.threshold = threshold

    def assess_response(
        self,
        response: str,
        question: Optional[str] = None,
        has_sources: bool = False,
        reasoning_steps: int = 0,
    ) -> ConfidenceAssessment:
        """
        Assess confidence in a response.

        Args:
            response: The agent's response text
            question: The original question (for complexity analysis)
            has_sources: Whether response cites sources
            reasoning_steps: Number of reasoning steps taken

        Returns:
            ConfidenceAssessment with detailed scoring
        """
        factors = []

        # Factor 1: Uncertainty language in response
        uncertainty_score, uncertainty_evidence = self._score_uncertainty_language(response)
        factors.append(ConfidenceFactorScore(
            factor=ConfidenceFactor.UNCERTAINTY_LANGUAGE,
            score=uncertainty_score,
            weight=0.30,
            explanation=f"Uncertainty markers detected: {len(uncertainty_evidence)}",
            evidence=uncertainty_evidence,
        ))

        # Factor 2: Certainty language in response
        certainty_score, certainty_evidence = self._score_certainty_language(response)
        factors.append(ConfidenceFactorScore(
            factor=ConfidenceFactor.CERTAINTY_LANGUAGE,
            score=certainty_score,
            weight=0.15,
            explanation=f"Certainty markers detected: {len(certainty_evidence)}",
            evidence=certainty_evidence,
        ))

        # Factor 3: Hedging phrases
        hedging_score, hedging_evidence = self._score_hedging(response)
        factors.append(ConfidenceFactorScore(
            factor=ConfidenceFactor.HEDGING_PHRASES,
            score=hedging_score,
            weight=0.15,
            explanation=f"Hedging phrases detected: {len(hedging_evidence)}",
            evidence=hedging_evidence,
        ))

        # Factor 4: Question complexity (if question provided)
        if question:
            complexity_score = self._score_question_complexity(question)
            factors.append(ConfidenceFactorScore(
                factor=ConfidenceFactor.QUESTION_COMPLEXITY,
                score=complexity_score,
                weight=0.15,
                explanation=f"Question complexity assessment",
            ))

            # Factor 5: Knowledge domain match
            domain_score = self._score_domain_match(question)
            factors.append(ConfidenceFactorScore(
                factor=ConfidenceFactor.KNOWLEDGE_DOMAIN,
                score=domain_score,
                weight=0.10,
                explanation=f"Domain confidence assessment",
            ))

        # Factor 6: Source availability
        source_score = 1.0 if has_sources else 0.7
        factors.append(ConfidenceFactorScore(
            factor=ConfidenceFactor.SOURCE_AVAILABILITY,
            score=source_score,
            weight=0.10,
            explanation="Sources cited" if has_sources else "No sources cited",
        ))

        # Factor 7: Response specificity
        specificity_score = self._score_specificity(response)
        factors.append(ConfidenceFactorScore(
            factor=ConfidenceFactor.SPECIFICITY,
            score=specificity_score,
            weight=0.05,
            explanation=f"Response specificity assessment",
        ))

        # Calculate weighted score
        raw_score = self._calculate_weighted_score(factors)

        # Determine confidence level
        level = self._score_to_level(raw_score)

        # Should we express uncertainty?
        should_express = raw_score < self.threshold

        # Should we ask for clarification?
        should_clarify = raw_score < 0.6

        # Get uncertainty phrases to use
        phrases = self._get_uncertainty_phrases(level)

        assessment = ConfidenceAssessment(
            raw_score=raw_score,
            level=level,
            factors=factors,
            should_express_uncertainty=should_express,
            should_ask_clarification=should_clarify,
            uncertainty_phrases=phrases,
            explanation=self._generate_explanation(raw_score, factors),
        )

        logger.debug(
            f"Confidence assessment: {level.value} ({raw_score:.2%}), "
            f"express_uncertainty={should_express}"
        )

        return assessment

    def _score_uncertainty_language(self, text: str) -> Tuple[float, List[str]]:
        """Score uncertainty language in text"""
        text_lower = text.lower()
        found = []
        min_score = 1.0

        for pattern, score, _ in self.UNCERTAINTY_MARKERS:
            if re.search(pattern, text_lower):
                found.append(re.search(pattern, text_lower).group())
                min_score = min(min_score, score)

        # More uncertainty markers = lower score
        if found:
            return min_score, found
        return 1.0, []

    def _score_certainty_language(self, text: str) -> Tuple[float, List[str]]:
        """Score certainty language in text"""
        text_lower = text.lower()
        found = []
        max_score = 0.5  # Start at baseline

        for pattern, score in self.CERTAINTY_MARKERS:
            if re.search(pattern, text_lower):
                found.append(re.search(pattern, text_lower).group())
                max_score = max(max_score, score)

        return max_score if found else 0.5, found

    def _score_hedging(self, text: str) -> Tuple[float, List[str]]:
        """Score hedging phrases in text"""
        text_lower = text.lower()
        found = []
        scores = []

        for pattern, score in self.HEDGING_PHRASES:
            if re.search(pattern, text_lower):
                found.append(re.search(pattern, text_lower).group())
                scores.append(score)

        if scores:
            return sum(scores) / len(scores), found
        return 1.0, []

    def _score_question_complexity(self, question: str) -> float:
        """Score question complexity (complex = lower base confidence)"""
        question_lower = question.lower()

        # Check for complex markers
        complex_count = sum(
            1 for pattern in self.COMPLEX_QUESTION_MARKERS
            if re.search(pattern, question_lower)
        )

        # Check for simple markers
        simple_count = sum(
            1 for pattern in self.SIMPLE_QUESTION_MARKERS
            if re.search(pattern, question_lower)
        )

        # More complex markers = lower confidence
        if complex_count > simple_count:
            return max(0.4, 0.8 - (complex_count * 0.1))
        elif simple_count > complex_count:
            return min(1.0, 0.8 + (simple_count * 0.05))
        return 0.75

    def _score_domain_match(self, question: str) -> float:
        """Score how well we know the domain"""
        question_lower = question.lower()

        # Check high confidence domains
        high_match = any(
            re.search(pattern, question_lower)
            for pattern in self.HIGH_CONFIDENCE_DOMAINS
        )

        # Check low confidence domains
        low_match = any(
            re.search(pattern, question_lower)
            for pattern in self.LOW_CONFIDENCE_DOMAINS
        )

        if high_match and not low_match:
            return 0.9
        elif low_match and not high_match:
            return 0.5
        elif high_match and low_match:
            return 0.7
        return 0.75  # Neutral

    def _score_specificity(self, response: str) -> float:
        """Score response specificity (specific = higher confidence)"""
        # Count specific elements
        has_numbers = bool(re.search(r'\d+', response))
        has_names = bool(re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?', response))
        has_dates = bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}', response))
        has_lists = bool(re.search(r'(?:^|\n)\s*[-•*\d+\.]', response))

        specificity_count = sum([has_numbers, has_names, has_dates, has_lists])

        if specificity_count >= 3:
            return 0.95
        elif specificity_count >= 2:
            return 0.85
        elif specificity_count >= 1:
            return 0.75
        return 0.6

    def _calculate_weighted_score(self, factors: List[ConfidenceFactorScore]) -> float:
        """Calculate weighted average of all factors"""
        total_weight = sum(f.weight for f in factors)
        weighted_sum = sum(f.score * f.weight for f in factors)

        if total_weight == 0:
            return self.base_confidence

        return weighted_sum / total_weight

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert numeric score to confidence level"""
        if score >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.85:
            return ConfidenceLevel.HIGH
        elif score >= 0.60:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.40:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _get_uncertainty_phrases(self, level: ConfidenceLevel) -> List[str]:
        """Get appropriate uncertainty phrases for the confidence level"""
        if level == ConfidenceLevel.VERY_LOW:
            return [
                "I'm not confident about this",
                "I'm uncertain",
                "I should clarify before proceeding",
            ]
        elif level == ConfidenceLevel.LOW:
            return [
                "I'm not entirely sure",
                "I believe, but I'm not certain",
                "This might not be accurate",
            ]
        elif level == ConfidenceLevel.MEDIUM:
            return [
                "I think",
                "As far as I know",
                "This should be correct, but please verify",
            ]
        elif level == ConfidenceLevel.HIGH:
            return [
                "I'm fairly confident",
                "Based on my understanding",
            ]
        else:  # VERY_HIGH
            return []  # No uncertainty needed

    def _generate_explanation(
        self,
        score: float,
        factors: List[ConfidenceFactorScore],
    ) -> str:
        """Generate human-readable explanation"""
        level = self._score_to_level(score)

        if level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH]:
            return f"High confidence ({score:.0%}) - Response is well-supported"
        elif level == ConfidenceLevel.MEDIUM:
            low_factors = [f for f in factors if f.score < 0.7]
            if low_factors:
                return f"Moderate confidence ({score:.0%}) - Some uncertainty in: {', '.join(f.factor.value for f in low_factors)}"
            return f"Moderate confidence ({score:.0%})"
        else:
            return f"Low confidence ({score:.0%}) - Consider asking for clarification"


# =============================================================================
# Quick Scoring Function (for backwards compatibility)
# =============================================================================

def assess_confidence(
    response: str,
    question: Optional[str] = None,
) -> Tuple[ConfidenceLevel, float]:
    """
    Quick function to assess confidence.

    This is a simple wrapper for backwards compatibility.

    Args:
        response: The agent's response
        question: The original question

    Returns:
        Tuple of (ConfidenceLevel, raw_score)
    """
    scorer = ConfidenceScorer()
    assessment = scorer.assess_response(response, question)
    return assessment.level, assessment.raw_score
