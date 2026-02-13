"""
=============================================================================
HUMMINGBIRD-LEA - Agentic Reasoning (Phase 2)
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
This module contains the ReAct reasoning loop, advanced ambiguity detection,
confidence scoring, clarifying questions engine, and reasoning transparency.

Components:
- ReAct Loop: Structured reasoning with Thought → Action → Observation
- Ambiguity Detection: Pattern-based detection of unclear requests
- Confidence Scoring: Multi-factor assessment of response confidence
- Clarifying Questions: Smart generation of contextual questions
- Reasoning Transparency: Making agent thinking visible to users
=============================================================================
"""

# ReAct Reasoning Loop
from .react import (
    ReActLoop,
    ReActPromptBuilder,
    ReasoningStep,
    ReasoningStepType,
    ReasoningAction,
    ReasoningResult,
    ReasoningStatus,
    ActionType,
)

# Advanced Ambiguity Detection
from .ambiguity import (
    AdvancedAmbiguityDetector,
    AmbiguityAnalysis,
    DetectedAmbiguity,
    AmbiguityType,
    AmbiguitySeverity,
    detect_ambiguity,
)

# Confidence Scoring
from .confidence import (
    ConfidenceScorer,
    ConfidenceAssessment,
    ConfidenceFactorScore,
    ConfidenceLevel,
    ConfidenceFactor,
    assess_confidence,
)

# Clarifying Questions Engine
from .clarifying import (
    ClarifyingQuestionsEngine,
    ClarifyingQuestion,
    ClarificationRequest,
    QuestionPriority,
    QuestionType,
    IntentCategory,
    generate_clarifying_questions,
)

# Reasoning Transparency
from .transparency import (
    ReasoningFormatter,
    TransparentResponse,
    ThinkingIndicator,
    TransparencyLevel,
    DisplayFormat,
    format_for_display,
)


# =============================================================================
# Convenience Classes for Integration
# =============================================================================

class ReasoningEngine:
    """
    Unified reasoning engine that combines all Phase 2 components.

    This is the main entry point for adding reasoning to agents.
    """

    def __init__(
        self,
        agent_name: str = "Agent",
        max_reasoning_steps: int = 10,
        confidence_threshold: float = 0.85,
        show_reasoning: bool = False,
        max_questions: int = 2,
    ):
        """
        Initialize the reasoning engine.

        Args:
            agent_name: Name of the agent (for personalization)
            max_reasoning_steps: Max steps in ReAct loop
            confidence_threshold: Below this, express uncertainty
            show_reasoning: Whether to show reasoning to users
            max_questions: Max clarifying questions per turn
        """
        self.agent_name = agent_name

        # Initialize components
        self.react_loop = ReActLoop(
            max_steps=max_reasoning_steps,
            confidence_threshold=confidence_threshold,
            show_reasoning=show_reasoning,
        )

        self.ambiguity_detector = AdvancedAmbiguityDetector(
            max_questions=max_questions,
        )

        self.confidence_scorer = ConfidenceScorer(
            threshold=confidence_threshold,
        )

        self.questions_engine = ClarifyingQuestionsEngine(
            agent_name=agent_name,
            max_questions=max_questions,
        )

        self.formatter = ReasoningFormatter()

        self.show_reasoning = show_reasoning

    def analyze_request(self, message: str, context: list = None) -> dict:
        """
        Analyze a user request before processing.

        Returns:
            Dict with ambiguity analysis and recommended questions
        """
        # Update detector with context
        self.ambiguity_detector.context_messages = context or []

        # Analyze for ambiguity
        ambiguity_analysis = self.ambiguity_detector.analyze(message)

        # Generate clarifying questions if needed
        clarification = None
        if ambiguity_analysis.needs_clarification:
            # Convert detected ambiguities to dict format for engine
            detected = [
                {"suggested_question": a.suggested_question}
                for a in ambiguity_analysis.ambiguities
            ]
            clarification = self.questions_engine.generate_questions(
                message,
                detected_ambiguities=detected,
                context=context,
            )

        return {
            "ambiguity": ambiguity_analysis,
            "needs_clarification": ambiguity_analysis.needs_clarification,
            "clarification": clarification,
            "ambiguity_score": ambiguity_analysis.ambiguity_score,
            "can_proceed": ambiguity_analysis.can_proceed_with_assumptions,
        }

    def assess_response(
        self,
        response: str,
        question: str = None,
        has_sources: bool = False,
    ) -> ConfidenceAssessment:
        """
        Assess confidence in a response.

        Returns:
            ConfidenceAssessment with detailed scoring
        """
        return self.confidence_scorer.assess_response(
            response=response,
            question=question,
            has_sources=has_sources,
            reasoning_steps=len(self.react_loop.steps),
        )

    def start_reasoning(self):
        """Start a new reasoning session"""
        self.react_loop.reset()

    def add_thought(self, thought: str, confidence: float = 1.0):
        """Add a thought to the reasoning chain"""
        self.react_loop.add_thought(thought, confidence)

    def add_reflection(self, reflection: str, confidence: float = 1.0):
        """Add a reflection to the reasoning chain"""
        self.react_loop.add_reflection(reflection, confidence)

    def complete_reasoning(
        self,
        response: str,
        clarifying_questions: list = None,
    ) -> ReasoningResult:
        """
        Complete the reasoning process.

        Returns:
            ReasoningResult with full reasoning chain
        """
        # Assess confidence
        assessment = self.assess_response(response)

        return self.react_loop.complete(
            response=response,
            confidence_score=assessment.raw_score,
            clarifying_questions=clarifying_questions,
        )

    def format_response(
        self,
        result: ReasoningResult,
        transparency_level: TransparencyLevel = None,
    ) -> str:
        """
        Format a reasoning result for display.

        Returns:
            Formatted response string
        """
        if transparency_level is None:
            transparency_level = (
                TransparencyLevel.SUMMARY if self.show_reasoning
                else TransparencyLevel.MINIMAL
            )

        transparent = self.formatter.format_reasoning_result(
            result,
            level=transparency_level,
        )

        return transparent.format(level=transparency_level)


# Export everything
__all__ = [
    # Main engine
    "ReasoningEngine",

    # ReAct
    "ReActLoop",
    "ReActPromptBuilder",
    "ReasoningStep",
    "ReasoningStepType",
    "ReasoningAction",
    "ReasoningResult",
    "ReasoningStatus",
    "ActionType",

    # Ambiguity
    "AdvancedAmbiguityDetector",
    "AmbiguityAnalysis",
    "DetectedAmbiguity",
    "AmbiguityType",
    "AmbiguitySeverity",
    "detect_ambiguity",

    # Confidence
    "ConfidenceScorer",
    "ConfidenceAssessment",
    "ConfidenceFactorScore",
    "ConfidenceLevel",
    "ConfidenceFactor",
    "assess_confidence",

    # Clarifying
    "ClarifyingQuestionsEngine",
    "ClarifyingQuestion",
    "ClarificationRequest",
    "QuestionPriority",
    "QuestionType",
    "IntentCategory",
    "generate_clarifying_questions",

    # Transparency
    "ReasoningFormatter",
    "TransparentResponse",
    "ThinkingIndicator",
    "TransparencyLevel",
    "DisplayFormat",
    "format_for_display",
]
