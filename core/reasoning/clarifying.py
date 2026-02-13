"""
=============================================================================
HUMMINGBIRD-LEA - Clarifying Questions Engine
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Smart generation of clarifying questions based on context and intent.

This engine:
1. Analyzes what information is missing
2. Generates contextual, helpful questions
3. Prioritizes questions by importance
4. Formats questions for natural conversation

Design Philosophy:
- Questions should feel natural, not robotic
- Offer options when possible
- Ask specific questions, not generic ones
- Respect the user's time (max 2-3 questions per turn)
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

class QuestionPriority(Enum):
    """Priority level for clarifying questions"""
    CRITICAL = 1    # Must ask before proceeding
    HIGH = 2        # Should ask, but could proceed with assumptions
    MEDIUM = 3      # Nice to know, improves response quality
    LOW = 4         # Optional enhancement


class QuestionType(Enum):
    """Types of clarifying questions"""
    WHAT = "what"           # What specifically?
    WHICH = "which"         # Which option/item?
    WHEN = "when"           # Time-related
    WHO = "who"             # Person/recipient
    WHERE = "where"         # Location/destination
    HOW = "how"             # Method/approach
    WHY = "why"             # Reason/purpose (use sparingly)
    CONFIRM = "confirm"     # Confirmation before action


class IntentCategory(Enum):
    """Categories of user intent"""
    ACTION_REQUEST = "action"       # User wants something done
    INFORMATION_REQUEST = "info"    # User wants to know something
    CREATION_REQUEST = "create"     # User wants something made
    MODIFICATION_REQUEST = "modify" # User wants something changed
    DELETION_REQUEST = "delete"     # User wants something removed
    COMMUNICATION_REQUEST = "comm"  # User wants to communicate
    SCHEDULING_REQUEST = "schedule" # User wants to schedule
    UNKNOWN = "unknown"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ClarifyingQuestion:
    """A generated clarifying question"""
    question: str
    question_type: QuestionType
    priority: QuestionPriority
    intent: IntentCategory
    options: Optional[List[str]] = None  # Suggested options if applicable
    context: Optional[str] = None        # Why we're asking
    follow_up: Optional[str] = None      # Follow-up if still unclear

    def format_with_options(self) -> str:
        """Format question with options if available"""
        if not self.options:
            return self.question

        formatted = self.question + "\n"
        for i, option in enumerate(self.options[:4], 1):  # Max 4 options
            formatted += f"  {i}. {option}\n"
        formatted += "  (or let me know if it's something else)"
        return formatted.strip()


@dataclass
class ClarificationRequest:
    """A complete clarification request to the user"""
    preamble: str                            # Friendly intro
    questions: List[ClarifyingQuestion]
    can_proceed_with_assumption: bool = False
    assumption: Optional[str] = None         # What we'd assume if not clarified
    closing: Optional[str] = None            # Friendly closing

    def format(self, max_questions: int = 2) -> str:
        """Format the complete clarification request"""
        parts = []

        # Preamble
        if self.preamble:
            parts.append(self.preamble)

        # Questions (limited)
        for q in self.questions[:max_questions]:
            if q.options:
                parts.append(q.format_with_options())
            else:
                parts.append(q.question)

        # Assumption notice
        if self.can_proceed_with_assumption and self.assumption:
            parts.append(f"\n(If you'd prefer, I can go ahead and assume {self.assumption})")

        # Closing
        if self.closing:
            parts.append(self.closing)

        return "\n\n".join(parts)


# =============================================================================
# Clarifying Questions Engine
# =============================================================================

class ClarifyingQuestionsEngine:
    """
    Generates contextual clarifying questions.

    This engine analyzes user messages and generates helpful,
    natural-sounding questions to gather missing information.
    """

    # -------------------------------------------------------------------------
    # Intent Detection Patterns
    # -------------------------------------------------------------------------
    INTENT_PATTERNS = {
        IntentCategory.COMMUNICATION_REQUEST: [
            r'\b(send|email|message|text|reply|forward)\b',
            r'\b(write|draft|compose)\s+(?:an?\s+)?(email|message|letter)\b',
        ],
        IntentCategory.SCHEDULING_REQUEST: [
            r'\b(schedule|book|set up|arrange|plan)\b',
            r'\b(meeting|call|appointment|event)\b',
        ],
        IntentCategory.CREATION_REQUEST: [
            r'\b(create|make|generate|build|produce)\b',
            r'\b(new|blank|fresh)\s+\w+\b',
        ],
        IntentCategory.MODIFICATION_REQUEST: [
            r'\b(update|modify|change|edit|revise|adjust)\b',
            r'\b(fix|correct|improve)\b',
        ],
        IntentCategory.DELETION_REQUEST: [
            r'\b(delete|remove|cancel|erase|drop)\b',
        ],
        IntentCategory.INFORMATION_REQUEST: [
            r'\b(what|who|where|when|why|how)\b.*\?',
            r'\b(tell me|show me|find|look up|search)\b',
            r'\b(explain|describe|summarize)\b',
        ],
    }

    # -------------------------------------------------------------------------
    # Question Templates by Intent
    # -------------------------------------------------------------------------
    QUESTION_TEMPLATES = {
        IntentCategory.COMMUNICATION_REQUEST: {
            "recipient": "Who should I send this to?",
            "content": "What would you like the message to say?",
            "subject": "What should the subject line be?",
            "urgency": "How urgent is this?",
            "tone": "Should this be formal or casual?",
        },
        IntentCategory.SCHEDULING_REQUEST: {
            "when": "When would you like to schedule this?",
            "who": "Who should be invited?",
            "duration": "How long should this be?",
            "location": "Where should this take place (or is it virtual)?",
            "recurring": "Is this a one-time or recurring event?",
        },
        IntentCategory.CREATION_REQUEST: {
            "what": "What exactly would you like me to create?",
            "format": "What format should this be in?",
            "content": "What should it include?",
            "purpose": "What will this be used for?",
        },
        IntentCategory.MODIFICATION_REQUEST: {
            "what": "What specifically should I change?",
            "target": "Which item should I modify?",
            "how": "How should I change it?",
        },
        IntentCategory.DELETION_REQUEST: {
            "confirm": "Are you sure you want to delete this? This cannot be undone.",
            "which": "Which item should I delete?",
        },
        IntentCategory.INFORMATION_REQUEST: {
            "scope": "What specifically would you like to know?",
            "context": "Could you give me more context?",
            "format": "How detailed would you like the answer?",
        },
    }

    # -------------------------------------------------------------------------
    # Missing Information Patterns
    # -------------------------------------------------------------------------
    MISSING_INFO_CHECKS = {
        "recipient_missing": (
            r'\b(send|email|message)\b(?!.*\b(to|@)\b)',
            IntentCategory.COMMUNICATION_REQUEST,
            "recipient"
        ),
        "time_missing": (
            r'\b(schedule|book|meeting)\b(?!.*\d)',
            IntentCategory.SCHEDULING_REQUEST,
            "when"
        ),
        "subject_missing": (
            r'\b(email|message)\b(?!.*\b(about|regarding|re:)\b)',
            IntentCategory.COMMUNICATION_REQUEST,
            "subject"
        ),
        "target_missing": (
            r'\b(delete|remove|update|modify)\b\s+(it|that|this)',
            IntentCategory.MODIFICATION_REQUEST,
            "target"
        ),
    }

    def __init__(
        self,
        agent_name: str = "Agent",
        max_questions: int = 2,
        friendly_tone: bool = True,
    ):
        """
        Initialize the engine.

        Args:
            agent_name: Name of the agent (for personalization)
            max_questions: Maximum questions to ask at once
            friendly_tone: Use friendly, conversational tone
        """
        self.agent_name = agent_name
        self.max_questions = max_questions
        self.friendly_tone = friendly_tone

    def generate_questions(
        self,
        message: str,
        detected_ambiguities: Optional[List[Dict]] = None,
        context: Optional[List[Dict]] = None,
    ) -> ClarificationRequest:
        """
        Generate clarifying questions for a message.

        Args:
            message: The user's message
            detected_ambiguities: Ambiguities detected by AmbiguityDetector
            context: Conversation context

        Returns:
            ClarificationRequest with prioritized questions
        """
        questions = []

        # Detect intent
        intent = self._detect_intent(message)

        # Check for missing critical information
        missing_info = self._check_missing_info(message, intent)
        for info_type, priority in missing_info:
            question = self._generate_question_for_info(intent, info_type, message)
            if question:
                questions.append(question)

        # Add questions from detected ambiguities
        if detected_ambiguities:
            for amb in detected_ambiguities:
                q = ClarifyingQuestion(
                    question=amb.get("suggested_question", "Could you clarify?"),
                    question_type=QuestionType.WHAT,
                    priority=QuestionPriority.HIGH,
                    intent=intent,
                )
                questions.append(q)

        # Sort by priority
        questions.sort(key=lambda q: q.priority.value)

        # Generate preamble
        preamble = self._generate_preamble(message, intent, len(questions))

        # Determine if we can proceed with assumption
        can_proceed, assumption = self._determine_assumption(message, intent, questions)

        return ClarificationRequest(
            preamble=preamble,
            questions=questions[:self.max_questions],
            can_proceed_with_assumption=can_proceed,
            assumption=assumption,
            closing=self._generate_closing() if self.friendly_tone else None,
        )

    def _detect_intent(self, message: str) -> IntentCategory:
        """Detect the user's intent from their message"""
        message_lower = message.lower()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return intent

        return IntentCategory.UNKNOWN

    def _check_missing_info(
        self,
        message: str,
        intent: IntentCategory,
    ) -> List[Tuple[str, QuestionPriority]]:
        """Check what critical information is missing"""
        missing = []
        message_lower = message.lower()

        for check_name, (pattern, check_intent, info_type) in self.MISSING_INFO_CHECKS.items():
            if intent == check_intent and re.search(pattern, message_lower):
                # Determine priority based on info type
                if info_type in ["recipient", "target", "confirm"]:
                    priority = QuestionPriority.CRITICAL
                elif info_type in ["when", "what"]:
                    priority = QuestionPriority.HIGH
                else:
                    priority = QuestionPriority.MEDIUM

                missing.append((info_type, priority))

        return missing

    def _generate_question_for_info(
        self,
        intent: IntentCategory,
        info_type: str,
        message: str,
    ) -> Optional[ClarifyingQuestion]:
        """Generate a specific question for missing information"""
        templates = self.QUESTION_TEMPLATES.get(intent, {})
        template = templates.get(info_type)

        if not template:
            return None

        # Determine question type
        if info_type in ["recipient", "who"]:
            q_type = QuestionType.WHO
        elif info_type in ["when", "time"]:
            q_type = QuestionType.WHEN
        elif info_type in ["where", "location"]:
            q_type = QuestionType.WHERE
        elif info_type in ["what", "content", "target"]:
            q_type = QuestionType.WHAT
        elif info_type in ["how", "format"]:
            q_type = QuestionType.HOW
        elif info_type == "confirm":
            q_type = QuestionType.CONFIRM
        else:
            q_type = QuestionType.WHAT

        # Determine priority
        if info_type in ["recipient", "confirm", "target"]:
            priority = QuestionPriority.CRITICAL
        elif info_type in ["when", "what", "content"]:
            priority = QuestionPriority.HIGH
        else:
            priority = QuestionPriority.MEDIUM

        # Generate options if applicable
        options = self._generate_options(intent, info_type, message)

        return ClarifyingQuestion(
            question=template,
            question_type=q_type,
            priority=priority,
            intent=intent,
            options=options,
        )

    def _generate_options(
        self,
        intent: IntentCategory,
        info_type: str,
        message: str,
    ) -> Optional[List[str]]:
        """Generate suggested options for a question"""
        # Only generate options for certain types
        if intent == IntentCategory.COMMUNICATION_REQUEST:
            if info_type == "tone":
                return ["Formal/Professional", "Casual/Friendly", "Brief/Direct"]
            elif info_type == "urgency":
                return ["Urgent (needs immediate attention)", "Normal priority", "Low priority (when convenient)"]

        elif intent == IntentCategory.SCHEDULING_REQUEST:
            if info_type == "duration":
                return ["15 minutes", "30 minutes", "1 hour", "Longer"]

        return None

    def _generate_preamble(
        self,
        message: str,
        intent: IntentCategory,
        question_count: int,
    ) -> str:
        """Generate a friendly preamble for the questions"""
        if not self.friendly_tone:
            return "I need some clarification:"

        if intent == IntentCategory.DELETION_REQUEST:
            return "Before I proceed with deleting anything, I want to make sure I understand correctly."

        if intent == IntentCategory.COMMUNICATION_REQUEST:
            return "I'd be happy to help with that! Just need a couple details:"

        if intent == IntentCategory.SCHEDULING_REQUEST:
            return "I can help set that up! Let me confirm a few things:"

        if question_count == 1:
            return "Quick question before I proceed:"
        else:
            return "I want to make sure I get this right. A couple of quick questions:"

    def _determine_assumption(
        self,
        message: str,
        intent: IntentCategory,
        questions: List[ClarifyingQuestion],
    ) -> Tuple[bool, Optional[str]]:
        """Determine if we can proceed with a reasonable assumption"""
        # Don't offer assumptions for critical actions
        if intent == IntentCategory.DELETION_REQUEST:
            return False, None

        # Don't offer assumptions if we have critical questions
        if any(q.priority == QuestionPriority.CRITICAL for q in questions):
            return False, None

        # For some intents, we can offer to proceed
        if intent == IntentCategory.INFORMATION_REQUEST:
            return True, "you want a general overview"

        return False, None

    def _generate_closing(self) -> str:
        """Generate a friendly closing"""
        closings = [
            "Let me know and I'll take care of it!",
            "Just let me know!",
            "Looking forward to helping!",
        ]
        import random
        return random.choice(closings)


# =============================================================================
# Quick Function (for simple use cases)
# =============================================================================

def generate_clarifying_questions(
    message: str,
    agent_name: str = "Agent",
    max_questions: int = 2,
) -> List[str]:
    """
    Quick function to generate clarifying questions.

    Args:
        message: The user's message
        agent_name: Name of the agent
        max_questions: Max questions to return

    Returns:
        List of question strings
    """
    engine = ClarifyingQuestionsEngine(
        agent_name=agent_name,
        max_questions=max_questions,
    )
    request = engine.generate_questions(message)
    return [q.question for q in request.questions]
