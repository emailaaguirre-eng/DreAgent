"""
=============================================================================
HUMMINGBIRD-LEA - Advanced Ambiguity Detection
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Advanced detection of ambiguous requests that need clarification.
This is KEY to reducing hallucinations - better to ask than to guess.

Features:
- Pattern-based detection with regex
- Context-aware analysis
- Prioritized question generation
- Ambiguity scoring
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

class AmbiguityType(Enum):
    """Types of ambiguity that can be detected"""
    VAGUE_REFERENCE = "vague_reference"      # "it", "that", "the file"
    INCOMPLETE_ACTION = "incomplete_action"   # "send" without what
    VAGUE_TIME = "vague_time"                # "soon", "later"
    MISSING_DETAILS = "missing_details"      # Critical info missing
    MULTIPLE_INTERPRETATIONS = "multiple_interpretations"  # Could mean several things
    IMPLICIT_ASSUMPTION = "implicit_assumption"  # Assumes context we don't have


class AmbiguitySeverity(Enum):
    """How critical is the ambiguity?"""
    LOW = 1       # Can probably proceed, but confirmation nice
    MEDIUM = 2    # Should clarify before proceeding
    HIGH = 3      # Must clarify - cannot proceed safely
    CRITICAL = 4  # Irreversible action - definitely must clarify


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DetectedAmbiguity:
    """A detected ambiguity in the user's request"""
    ambiguity_type: AmbiguityType
    severity: AmbiguitySeverity
    trigger: str                    # What text triggered detection
    description: str                # Human-readable description
    suggested_question: str         # Question to ask user
    context_hint: Optional[str] = None  # Additional context if available

    @property
    def priority(self) -> int:
        """Higher priority = should ask first"""
        return self.severity.value


@dataclass
class AmbiguityAnalysis:
    """Complete analysis of ambiguities in a request"""
    original_message: str
    ambiguities: List[DetectedAmbiguity] = field(default_factory=list)
    ambiguity_score: float = 0.0  # 0.0 = crystal clear, 1.0 = very ambiguous
    needs_clarification: bool = False
    recommended_questions: List[str] = field(default_factory=list)
    can_proceed_with_assumptions: bool = True

    def get_top_questions(self, max_questions: int = 2) -> List[str]:
        """Get the most important clarifying questions"""
        # Sort by priority (severity), take top N
        sorted_ambiguities = sorted(
            self.ambiguities,
            key=lambda a: a.priority,
            reverse=True
        )
        return [a.suggested_question for a in sorted_ambiguities[:max_questions]]


# =============================================================================
# Advanced Ambiguity Detector
# =============================================================================

class AdvancedAmbiguityDetector:
    """
    Advanced ambiguity detection with pattern matching and context awareness.

    Design Philosophy:
    - It's better to ask a clarifying question than to hallucinate
    - Users prefer being asked vs receiving wrong information
    - Critical/irreversible actions MUST be confirmed
    """

    # -------------------------------------------------------------------------
    # Vague References (things that need a specific referent)
    # -------------------------------------------------------------------------
    VAGUE_REFERENCE_PATTERNS = [
        # Pronouns without clear antecedent
        (r'\b(it|that|this|those|these)\b(?!\s+is\s+a\s+)', 'pronoun'),
        (r'\bthe (file|email|document|report|project|meeting|data|thing|item|one)\b', 'definite_article'),
        (r'\b(they|them|their)\b', 'plural_pronoun'),
        (r'\b(he|she|him|her)\b', 'person_pronoun'),
    ]

    # -------------------------------------------------------------------------
    # Incomplete Actions (verbs that need objects/details)
    # -------------------------------------------------------------------------
    INCOMPLETE_ACTION_PATTERNS = [
        # Common actions that need an object
        (r'^(send|create|update|delete|move|copy|check|review)\s*$', 'action_no_object'),
        (r'^(send|create|update|delete|move|copy|check|review)\s+(?:it|that|this)\s*$', 'action_vague_object'),
        (r'^(fix|change|modify|edit)\s*(?:it|that|this)?\s*$', 'modification_vague'),
        (r'^(do|handle|process|complete)\s+(?:it|that|this|the\s+\w+)\s*$', 'generic_action'),
    ]

    # -------------------------------------------------------------------------
    # Vague Time References
    # -------------------------------------------------------------------------
    VAGUE_TIME_PATTERNS = [
        (r'\b(soon|later|eventually|sometime|someday)\b', 'indefinite_time'),
        (r'\b(the other day|recently|a while ago|back then)\b', 'vague_past'),
        (r'\b(tomorrow|next week|next month)\b', 'relative_future'),
        (r'\bwhen (possible|you can|you have time)\b', 'conditional_time'),
        (r'\b(asap|as soon as possible|urgent|urgently)\b', 'urgency_undefined'),
    ]

    # -------------------------------------------------------------------------
    # Missing Critical Details (for specific action types)
    # -------------------------------------------------------------------------
    EMAIL_PATTERNS = [
        (r'\b(send|email|mail)\b.*(?!.*(@|to\s+\w+))', 'email_no_recipient'),
    ]

    MEETING_PATTERNS = [
        (r'\b(schedule|book|set up)\s+(a\s+)?(meeting|call)\b(?!.*\d)', 'meeting_no_time'),
    ]

    FILE_PATTERNS = [
        (r'\b(save|create|write)\s+(a\s+)?(file|document)\b(?!.*(named|called|as\s+))', 'file_no_name'),
    ]

    # -------------------------------------------------------------------------
    # Irreversible Actions (MUST confirm)
    # -------------------------------------------------------------------------
    CRITICAL_ACTION_PATTERNS = [
        (r'\b(delete|remove|erase|destroy|drop)\b', 'destructive_action'),
        (r'\b(send|submit|publish|post|deploy)\b', 'irreversible_send'),
        (r'\b(cancel|terminate|end)\b', 'termination_action'),
        (r'\b(pay|transfer|wire|charge)\b', 'financial_action'),
    ]

    def __init__(
        self,
        context_messages: Optional[List[Dict]] = None,
        max_questions: int = 2,
    ):
        """
        Initialize the detector.

        Args:
            context_messages: Previous messages in conversation (for context)
            max_questions: Maximum clarifying questions to return
        """
        self.context_messages = context_messages or []
        self.max_questions = max_questions

    def analyze(self, message: str) -> AmbiguityAnalysis:
        """
        Analyze a message for ambiguities.

        Args:
            message: The user's message to analyze

        Returns:
            AmbiguityAnalysis with detected issues and recommended questions
        """
        ambiguities = []
        message_lower = message.lower().strip()

        # Skip very short messages or greetings
        if len(message_lower.split()) < 2:
            return AmbiguityAnalysis(
                original_message=message,
                ambiguity_score=0.0,
                needs_clarification=False,
            )

        # Check each pattern category
        ambiguities.extend(self._check_vague_references(message_lower, message))
        ambiguities.extend(self._check_incomplete_actions(message_lower, message))
        ambiguities.extend(self._check_vague_time(message_lower, message))
        ambiguities.extend(self._check_missing_details(message_lower, message))
        ambiguities.extend(self._check_critical_actions(message_lower, message))

        # Filter based on context (if we have enough conversation history)
        if self.context_messages:
            ambiguities = self._filter_by_context(ambiguities)

        # Calculate overall ambiguity score
        ambiguity_score = self._calculate_score(ambiguities)

        # Determine if clarification is needed
        needs_clarification = (
            ambiguity_score > 0.3 or
            any(a.severity == AmbiguitySeverity.CRITICAL for a in ambiguities) or
            any(a.severity == AmbiguitySeverity.HIGH for a in ambiguities)
        )

        # Can we proceed with reasonable assumptions?
        can_proceed = (
            ambiguity_score < 0.5 and
            not any(a.severity == AmbiguitySeverity.CRITICAL for a in ambiguities)
        )

        # Get recommended questions
        analysis = AmbiguityAnalysis(
            original_message=message,
            ambiguities=ambiguities,
            ambiguity_score=ambiguity_score,
            needs_clarification=needs_clarification,
            recommended_questions=self._get_questions(ambiguities),
            can_proceed_with_assumptions=can_proceed,
        )

        if ambiguities:
            logger.info(
                f"Ambiguity detected: score={ambiguity_score:.2f}, "
                f"issues={len(ambiguities)}, needs_clarification={needs_clarification}"
            )

        return analysis

    def _check_vague_references(
        self,
        message_lower: str,
        original: str,
    ) -> List[DetectedAmbiguity]:
        """Check for vague references like 'it', 'that', 'the file'"""
        ambiguities = []

        for pattern, pattern_type in self.VAGUE_REFERENCE_PATTERNS:
            matches = re.finditer(pattern, message_lower, re.IGNORECASE)
            for match in matches:
                trigger = match.group(0)

                # Skip if message is long enough to likely have context
                if len(message_lower.split()) > 15 and pattern_type == 'pronoun':
                    continue

                question = self._generate_reference_question(trigger, pattern_type)

                ambiguities.append(DetectedAmbiguity(
                    ambiguity_type=AmbiguityType.VAGUE_REFERENCE,
                    severity=AmbiguitySeverity.MEDIUM,
                    trigger=trigger,
                    description=f"Vague reference '{trigger}' needs clarification",
                    suggested_question=question,
                ))
                break  # One per pattern type

        return ambiguities

    def _check_incomplete_actions(
        self,
        message_lower: str,
        original: str,
    ) -> List[DetectedAmbiguity]:
        """Check for incomplete action commands"""
        ambiguities = []

        for pattern, pattern_type in self.INCOMPLETE_ACTION_PATTERNS:
            if re.search(pattern, message_lower, re.IGNORECASE):
                match = re.search(r'^(\w+)', message_lower)
                action = match.group(1) if match else "do"

                ambiguities.append(DetectedAmbiguity(
                    ambiguity_type=AmbiguityType.INCOMPLETE_ACTION,
                    severity=AmbiguitySeverity.HIGH,
                    trigger=message_lower[:50],
                    description=f"Action '{action}' is missing details",
                    suggested_question=f"What specifically would you like me to {action}?",
                ))
                break

        return ambiguities

    def _check_vague_time(
        self,
        message_lower: str,
        original: str,
    ) -> List[DetectedAmbiguity]:
        """Check for vague time references"""
        ambiguities = []

        for pattern, pattern_type in self.VAGUE_TIME_PATTERNS:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                trigger = match.group(0)

                # Different severity based on type
                if pattern_type == 'urgency_undefined':
                    severity = AmbiguitySeverity.MEDIUM
                    question = f"When you say '{trigger}', what's your actual deadline?"
                elif pattern_type == 'relative_future':
                    severity = AmbiguitySeverity.LOW
                    question = f"Could you specify an exact date/time for '{trigger}'?"
                else:
                    severity = AmbiguitySeverity.LOW
                    question = f"Could you be more specific about the timing ('{trigger}')?"

                ambiguities.append(DetectedAmbiguity(
                    ambiguity_type=AmbiguityType.VAGUE_TIME,
                    severity=severity,
                    trigger=trigger,
                    description=f"Time reference '{trigger}' is vague",
                    suggested_question=question,
                ))
                break

        return ambiguities

    def _check_missing_details(
        self,
        message_lower: str,
        original: str,
    ) -> List[DetectedAmbiguity]:
        """Check for missing critical details in specific action types"""
        ambiguities = []

        # Check email-related requests
        if any(word in message_lower for word in ['email', 'send', 'mail']):
            if not re.search(r'@|to\s+\w+', message_lower):
                if 'draft' not in message_lower:  # Drafts don't need recipients yet
                    ambiguities.append(DetectedAmbiguity(
                        ambiguity_type=AmbiguityType.MISSING_DETAILS,
                        severity=AmbiguitySeverity.HIGH,
                        trigger="email/send",
                        description="Email action missing recipient",
                        suggested_question="Who should I send this to?",
                    ))

        # Check meeting-related requests
        if any(word in message_lower for word in ['meeting', 'schedule', 'call', 'book']):
            if not re.search(r'\d{1,2}[:/]\d{2}|\d{1,2}\s*(am|pm)|monday|tuesday|wednesday|thursday|friday', message_lower):
                ambiguities.append(DetectedAmbiguity(
                    ambiguity_type=AmbiguityType.MISSING_DETAILS,
                    severity=AmbiguitySeverity.MEDIUM,
                    trigger="meeting/schedule",
                    description="Meeting missing time details",
                    suggested_question="When would you like to schedule this?",
                ))

        return ambiguities

    def _check_critical_actions(
        self,
        message_lower: str,
        original: str,
    ) -> List[DetectedAmbiguity]:
        """Check for critical/irreversible actions that need confirmation"""
        ambiguities = []

        for pattern, pattern_type in self.CRITICAL_ACTION_PATTERNS:
            match = re.search(pattern, message_lower, re.IGNORECASE)
            if match:
                trigger = match.group(0)

                # Check if there's already explicit confirmation language
                if any(word in message_lower for word in ['confirm', 'yes', 'definitely', 'for sure']):
                    continue

                if pattern_type == 'destructive_action':
                    question = f"Before I {trigger} anything, could you confirm exactly what should be {trigger}d?"
                    severity = AmbiguitySeverity.CRITICAL
                elif pattern_type == 'financial_action':
                    question = f"This involves a financial action. Can you confirm the exact details?"
                    severity = AmbiguitySeverity.CRITICAL
                else:
                    question = f"This action cannot be undone. Are you sure you want to proceed?"
                    severity = AmbiguitySeverity.HIGH

                ambiguities.append(DetectedAmbiguity(
                    ambiguity_type=AmbiguityType.IMPLICIT_ASSUMPTION,
                    severity=severity,
                    trigger=trigger,
                    description=f"Critical action '{trigger}' needs confirmation",
                    suggested_question=question,
                ))
                break

        return ambiguities

    def _filter_by_context(
        self,
        ambiguities: List[DetectedAmbiguity],
    ) -> List[DetectedAmbiguity]:
        """Filter ambiguities based on conversation context"""
        if not self.context_messages or len(self.context_messages) < 2:
            return ambiguities

        # Build context string from recent messages
        recent_context = " ".join(
            msg.get("content", "") for msg in self.context_messages[-5:]
        ).lower()

        filtered = []
        for amb in ambiguities:
            # If a vague reference was likely defined in recent context, skip it
            if amb.ambiguity_type == AmbiguityType.VAGUE_REFERENCE:
                trigger = amb.trigger.lower()
                # Check if there's a recent definition
                if trigger in recent_context and recent_context.count(trigger) > 1:
                    continue  # Probably defined in context, skip

            filtered.append(amb)

        return filtered

    def _calculate_score(self, ambiguities: List[DetectedAmbiguity]) -> float:
        """Calculate overall ambiguity score (0.0 to 1.0)"""
        if not ambiguities:
            return 0.0

        # Weight by severity
        weights = {
            AmbiguitySeverity.LOW: 0.15,
            AmbiguitySeverity.MEDIUM: 0.30,
            AmbiguitySeverity.HIGH: 0.50,
            AmbiguitySeverity.CRITICAL: 0.80,
        }

        total_weight = sum(weights[a.severity] for a in ambiguities)

        # Normalize to 0-1 range (cap at 1.0)
        return min(1.0, total_weight)

    def _generate_reference_question(self, trigger: str, pattern_type: str) -> str:
        """Generate a contextual question for vague references"""
        if pattern_type == 'definite_article':
            return f"When you say '{trigger}', which specific one are you referring to?"
        elif pattern_type == 'pronoun':
            return f"Could you clarify what '{trigger}' refers to?"
        elif pattern_type == 'plural_pronoun':
            return f"Who are you referring to with '{trigger}'?"
        elif pattern_type == 'person_pronoun':
            return f"Who is '{trigger}' in this context?"
        else:
            return f"Could you be more specific about '{trigger}'?"

    def _get_questions(self, ambiguities: List[DetectedAmbiguity]) -> List[str]:
        """Get the most important clarifying questions"""
        # Sort by priority
        sorted_ambiguities = sorted(
            ambiguities,
            key=lambda a: a.priority,
            reverse=True
        )

        # Get unique questions (some might be duplicates)
        seen = set()
        questions = []
        for amb in sorted_ambiguities:
            if amb.suggested_question not in seen:
                questions.append(amb.suggested_question)
                seen.add(amb.suggested_question)
            if len(questions) >= self.max_questions:
                break

        return questions


# =============================================================================
# Quick Detection Function (for backwards compatibility)
# =============================================================================

def detect_ambiguity(message: str, context: Optional[List[Dict]] = None) -> List[str]:
    """
    Quick function to detect ambiguity and return clarifying questions.

    This is a simple wrapper for backwards compatibility with the basic
    AmbiguityDetector.detect() method.

    Args:
        message: The user's message
        context: Optional conversation context

    Returns:
        List of clarifying questions (max 2)
    """
    detector = AdvancedAmbiguityDetector(context_messages=context)
    analysis = detector.analyze(message)
    return analysis.get_top_questions(max_questions=2)
