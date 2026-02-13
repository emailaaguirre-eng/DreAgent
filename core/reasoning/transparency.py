"""
=============================================================================
HUMMINGBIRD-LEA - Reasoning Transparency
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Makes the agent's reasoning visible and understandable to users.

Features:
1. Formats reasoning steps for display
2. Generates thinking indicators ("I'm considering...")
3. Creates transparent explanations of decisions
4. Helps users understand why the agent is asking questions

Design Philosophy:
- Transparency builds trust
- Users should understand WHY the agent is uncertain
- Showing reasoning helps users provide better input
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

from .react import ReasoningStep, ReasoningStepType, ReasoningResult
from .confidence import ConfidenceAssessment, ConfidenceLevel

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TransparencyLevel(Enum):
    """How much reasoning to show"""
    NONE = "none"           # Show nothing (just the answer)
    MINIMAL = "minimal"     # Show brief thinking indicator
    SUMMARY = "summary"     # Show summary of reasoning
    DETAILED = "detailed"   # Show full reasoning steps
    DEBUG = "debug"         # Show everything (for development)


class DisplayFormat(Enum):
    """Format for displaying reasoning"""
    PLAIN_TEXT = "plain"      # Simple text
    MARKDOWN = "markdown"     # Markdown formatting
    STRUCTURED = "structured" # JSON-like structure


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ThinkingIndicator:
    """A thinking indicator to show during processing"""
    message: str
    phase: str  # What phase we're in
    progress: Optional[float] = None  # 0.0 to 1.0 if known
    is_final: bool = False


@dataclass
class TransparentResponse:
    """A response with optional reasoning visibility"""
    content: str                           # The main response
    reasoning_visible: bool = False        # Whether to show reasoning
    thinking_summary: Optional[str] = None # Brief summary if not showing full reasoning
    full_reasoning: Optional[str] = None   # Full reasoning if visible
    confidence_note: Optional[str] = None  # Note about confidence level
    uncertainty_explanation: Optional[str] = None  # Why we're uncertain

    def format(self, level: TransparencyLevel = TransparencyLevel.MINIMAL) -> str:
        """Format the response based on transparency level"""
        parts = []

        # Add reasoning based on level
        if level == TransparencyLevel.DEBUG and self.full_reasoning:
            parts.append(f"[DEBUG] Reasoning:\n{self.full_reasoning}\n")
        elif level == TransparencyLevel.DETAILED and self.full_reasoning:
            parts.append(f"Here's my thinking:\n{self.full_reasoning}\n")
        elif level == TransparencyLevel.SUMMARY and self.thinking_summary:
            parts.append(f"*{self.thinking_summary}*\n")

        # Add main content
        parts.append(self.content)

        # Add uncertainty explanation if relevant
        if self.uncertainty_explanation and level != TransparencyLevel.NONE:
            parts.append(f"\n\n*{self.uncertainty_explanation}*")

        # Add confidence note if relevant and not NONE level
        if self.confidence_note and level != TransparencyLevel.NONE:
            parts.append(f"\n\n{self.confidence_note}")

        return "\n".join(parts)


# =============================================================================
# Reasoning Formatter
# =============================================================================

class ReasoningFormatter:
    """
    Formats reasoning for display to users.
    """

    # Thought indicators by category
    THOUGHT_PREFIXES = {
        "analyzing": ["Analyzing", "Looking at", "Considering"],
        "checking": ["Checking", "Verifying", "Making sure"],
        "planning": ["Planning", "Figuring out", "Deciding"],
        "uncertain": ["Hmm, I'm thinking about", "Let me consider", "I'm not sure about"],
    }

    # Confidence descriptions
    CONFIDENCE_DESCRIPTIONS = {
        ConfidenceLevel.VERY_HIGH: None,  # Don't mention high confidence
        ConfidenceLevel.HIGH: None,
        ConfidenceLevel.MEDIUM: "I'm fairly confident about this, but let me know if something seems off.",
        ConfidenceLevel.LOW: "I'm not entirely sure about this. Please verify the details.",
        ConfidenceLevel.VERY_LOW: "I'm quite uncertain about this. We should probably clarify before proceeding.",
    }

    def __init__(
        self,
        default_level: TransparencyLevel = TransparencyLevel.MINIMAL,
        format_type: DisplayFormat = DisplayFormat.PLAIN_TEXT,
    ):
        """
        Initialize the formatter.

        Args:
            default_level: Default transparency level
            format_type: Output format type
        """
        self.default_level = default_level
        self.format_type = format_type

    def format_reasoning_result(
        self,
        result: ReasoningResult,
        level: Optional[TransparencyLevel] = None,
    ) -> TransparentResponse:
        """
        Format a ReasoningResult for display.

        Args:
            result: The ReasoningResult to format
            level: Transparency level (uses default if not specified)

        Returns:
            TransparentResponse ready for display
        """
        level = level or self.default_level

        # Generate thinking summary
        thinking_summary = self._generate_summary(result.steps)

        # Generate full reasoning
        full_reasoning = self._format_steps(result.steps)

        # Generate confidence note if needed
        confidence_note = None
        if result.confidence_score < 0.85:
            confidence_note = self._generate_confidence_note(result.confidence_score)

        # Generate uncertainty explanation if we're asking questions
        uncertainty_explanation = None
        if result.clarifying_questions:
            uncertainty_explanation = self._explain_uncertainty(result)

        return TransparentResponse(
            content=result.response or "",
            reasoning_visible=level in [TransparencyLevel.DETAILED, TransparencyLevel.DEBUG],
            thinking_summary=thinking_summary,
            full_reasoning=full_reasoning,
            confidence_note=confidence_note,
            uncertainty_explanation=uncertainty_explanation,
        )

    def format_confidence_assessment(
        self,
        assessment: ConfidenceAssessment,
    ) -> Optional[str]:
        """
        Format a confidence assessment for user display.

        Returns None if confidence is high enough to not mention.
        """
        description = self.CONFIDENCE_DESCRIPTIONS.get(assessment.level)
        if not description:
            return None

        return description

    def generate_thinking_indicator(
        self,
        phase: str,
        step_number: int = 0,
        total_steps: Optional[int] = None,
    ) -> ThinkingIndicator:
        """
        Generate a thinking indicator for real-time display.

        Args:
            phase: Current phase (analyzing, checking, planning, uncertain)
            step_number: Current step number
            total_steps: Total expected steps (if known)

        Returns:
            ThinkingIndicator for display
        """
        import random

        prefixes = self.THOUGHT_PREFIXES.get(phase, self.THOUGHT_PREFIXES["analyzing"])
        prefix = random.choice(prefixes)

        messages = {
            "analyzing": f"{prefix} your request...",
            "checking": f"{prefix} the details...",
            "planning": f"{prefix} the best approach...",
            "uncertain": f"{prefix} this carefully...",
        }

        message = messages.get(phase, f"{prefix}...")

        progress = None
        if total_steps and total_steps > 0:
            progress = step_number / total_steps

        return ThinkingIndicator(
            message=message,
            phase=phase,
            progress=progress,
            is_final=(step_number == total_steps if total_steps else False),
        )

    def _generate_summary(self, steps: List[ReasoningStep]) -> str:
        """Generate a brief summary of reasoning steps"""
        thoughts = [s.content for s in steps if s.step_type == ReasoningStepType.THOUGHT]
        reflections = [s.content for s in steps if s.step_type == ReasoningStepType.REFLECTION]

        if reflections:
            return f"After thinking about this: {reflections[-1][:100]}"
        elif thoughts:
            return f"Considered: {thoughts[0][:100]}"
        else:
            return "Processed request"

    def _format_steps(self, steps: List[ReasoningStep]) -> str:
        """Format all reasoning steps"""
        if self.format_type == DisplayFormat.MARKDOWN:
            return self._format_steps_markdown(steps)
        elif self.format_type == DisplayFormat.STRUCTURED:
            return self._format_steps_structured(steps)
        else:
            return self._format_steps_plain(steps)

    def _format_steps_plain(self, steps: List[ReasoningStep]) -> str:
        """Format steps as plain text"""
        lines = []
        for i, step in enumerate(steps, 1):
            prefix = step.step_type.value.capitalize()
            lines.append(f"{i}. {prefix}: {step.content}")
        return "\n".join(lines)

    def _format_steps_markdown(self, steps: List[ReasoningStep]) -> str:
        """Format steps as markdown"""
        lines = ["### Reasoning Steps\n"]
        for step in steps:
            emoji = {
                ReasoningStepType.THOUGHT: "💭",
                ReasoningStepType.ACTION: "⚡",
                ReasoningStepType.OBSERVATION: "👁️",
                ReasoningStepType.REFLECTION: "🤔",
                ReasoningStepType.QUESTION: "❓",
            }.get(step.step_type, "•")

            lines.append(f"{emoji} **{step.step_type.value.capitalize()}**: {step.content}")
        return "\n".join(lines)

    def _format_steps_structured(self, steps: List[ReasoningStep]) -> str:
        """Format steps as structured text"""
        lines = []
        for step in steps:
            lines.append(f"[{step.step_type.value.upper()}] {step.content}")
        return "\n".join(lines)

    def _generate_confidence_note(self, score: float) -> str:
        """Generate a confidence note based on score"""
        percentage = int(score * 100)

        if score < 0.4:
            return f"⚠️ Low confidence ({percentage}%): I'm quite uncertain about this and recommend we clarify."
        elif score < 0.6:
            return f"Note: I'm only moderately confident ({percentage}%) about this response."
        elif score < 0.85:
            return f"Note: My confidence is {percentage}% - please verify the key details."
        else:
            return ""

    def _explain_uncertainty(self, result: ReasoningResult) -> str:
        """Explain why we're asking for clarification"""
        if not result.clarifying_questions:
            return ""

        if len(result.clarifying_questions) == 1:
            return "I want to make sure I understand correctly, so I have a quick question."
        else:
            return f"I want to make sure I get this right, so I have {len(result.clarifying_questions)} quick questions."


# =============================================================================
# Helper Functions
# =============================================================================

def format_for_display(
    response: str,
    reasoning_steps: Optional[List[ReasoningStep]] = None,
    confidence: Optional[ConfidenceAssessment] = None,
    level: TransparencyLevel = TransparencyLevel.MINIMAL,
) -> str:
    """
    Quick function to format a response with transparency.

    Args:
        response: The main response text
        reasoning_steps: Optional reasoning steps
        confidence: Optional confidence assessment
        level: Transparency level

    Returns:
        Formatted response string
    """
    formatter = ReasoningFormatter(default_level=level)

    parts = []

    # Add reasoning summary if available and level allows
    if reasoning_steps and level != TransparencyLevel.NONE:
        summary = formatter._generate_summary(reasoning_steps)
        if level == TransparencyLevel.DETAILED:
            full = formatter._format_steps(reasoning_steps)
            parts.append(f"My reasoning:\n{full}\n")
        elif level == TransparencyLevel.SUMMARY:
            parts.append(f"*{summary}*\n")

    # Add main response
    parts.append(response)

    # Add confidence note if needed
    if confidence and confidence.should_express_uncertainty:
        note = formatter.format_confidence_assessment(confidence)
        if note:
            parts.append(f"\n\n{note}")

    return "\n".join(parts)
