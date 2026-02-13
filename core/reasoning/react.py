"""
=============================================================================
HUMMINGBIRD-LEA - ReAct Reasoning Loop
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Implementation of the ReAct (Reasoning + Acting) pattern for agentic behavior.

The ReAct loop follows this pattern:
1. THOUGHT - Agent reasons about what to do
2. ACTION - Agent decides on an action (or asks for clarification)
3. OBSERVATION - Agent observes the result
4. REFLECTION - Agent reflects on whether goal is achieved

This enables step-by-step reasoning with transparency.
=============================================================================
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class ReasoningStepType(Enum):
    """Types of steps in the reasoning process"""
    THOUGHT = "thought"       # Internal reasoning
    ACTION = "action"         # Decision or action to take
    OBSERVATION = "observation"  # Result of an action
    REFLECTION = "reflection"    # Self-assessment
    QUESTION = "question"     # Clarifying question for user


class ActionType(Enum):
    """Types of actions the agent can take"""
    RESPOND = "respond"           # Provide a response to user
    CLARIFY = "clarify"           # Ask for clarification
    SEARCH = "search"             # Search for information (future)
    EXECUTE = "execute"           # Execute a command (future)
    DELEGATE = "delegate"         # Hand off to another agent (future)
    WAIT = "wait"                 # Wait for more information


class ReasoningStatus(Enum):
    """Status of the reasoning process"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    NEEDS_CLARIFICATION = "needs_clarification"
    BLOCKED = "blocked"
    FAILED = "failed"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ReasoningStep:
    """A single step in the reasoning process"""
    step_type: ReasoningStepType
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_display(self, show_timestamps: bool = False) -> str:
        """Format step for display to user"""
        prefix = {
            ReasoningStepType.THOUGHT: "Thinking",
            ReasoningStepType.ACTION: "Action",
            ReasoningStepType.OBSERVATION: "Observed",
            ReasoningStepType.REFLECTION: "Reflecting",
            ReasoningStepType.QUESTION: "Question",
        }[self.step_type]

        if show_timestamps:
            time_str = self.timestamp.strftime("%H:%M:%S")
            return f"[{time_str}] {prefix}: {self.content}"
        return f"{prefix}: {self.content}"


@dataclass
class ReasoningAction:
    """An action decided upon during reasoning"""
    action_type: ActionType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_confirmation: bool = False


@dataclass
class ReasoningResult:
    """Final result of the reasoning process"""
    status: ReasoningStatus
    steps: List[ReasoningStep]
    final_action: Optional[ReasoningAction] = None
    response: Optional[str] = None
    clarifying_questions: Optional[List[str]] = None
    confidence_score: float = 1.0
    reasoning_visible: bool = False  # Whether to show reasoning to user
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def thoughts(self) -> List[str]:
        """Get all thought steps as strings"""
        return [s.content for s in self.steps if s.step_type == ReasoningStepType.THOUGHT]

    @property
    def formatted_reasoning(self) -> str:
        """Get formatted reasoning for display"""
        if not self.reasoning_visible:
            return ""

        lines = []
        for step in self.steps:
            if step.step_type in [ReasoningStepType.THOUGHT, ReasoningStepType.REFLECTION]:
                lines.append(step.to_display())
        return "\n".join(lines)


# =============================================================================
# ReAct Loop
# =============================================================================

class ReActLoop:
    """
    The ReAct reasoning loop.

    Implements a structured approach to agent reasoning:
    1. Analyze the user's request
    2. Think through the approach
    3. Decide on an action
    4. Reflect on whether clarification is needed

    This helps prevent hallucination by making the reasoning explicit.
    """

    def __init__(
        self,
        max_steps: int = 10,
        confidence_threshold: float = 0.85,
        show_reasoning: bool = False,
    ):
        """
        Initialize the ReAct loop.

        Args:
            max_steps: Maximum reasoning steps before forcing a decision
            confidence_threshold: Minimum confidence to proceed without clarification
            show_reasoning: Whether to expose reasoning to the user
        """
        self.max_steps = max_steps
        self.confidence_threshold = confidence_threshold
        self.show_reasoning = show_reasoning
        self.steps: List[ReasoningStep] = []
        self.status = ReasoningStatus.IN_PROGRESS

    def reset(self):
        """Reset the loop for a new reasoning session"""
        self.steps = []
        self.status = ReasoningStatus.IN_PROGRESS

    def add_thought(
        self,
        content: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStep:
        """Add a thought step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.THOUGHT,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.steps.append(step)
        logger.debug(f"ReAct Thought: {content}")
        return step

    def add_observation(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStep:
        """Add an observation step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.OBSERVATION,
            content=content,
            metadata=metadata or {},
        )
        self.steps.append(step)
        logger.debug(f"ReAct Observation: {content}")
        return step

    def add_reflection(
        self,
        content: str,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStep:
        """Add a reflection step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.REFLECTION,
            content=content,
            confidence=confidence,
            metadata=metadata or {},
        )
        self.steps.append(step)
        logger.debug(f"ReAct Reflection: {content}")
        return step

    def add_question(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningStep:
        """Add a clarifying question step"""
        step = ReasoningStep(
            step_type=ReasoningStepType.QUESTION,
            content=content,
            metadata=metadata or {},
        )
        self.steps.append(step)
        logger.debug(f"ReAct Question: {content}")
        return step

    def decide_action(
        self,
        action_type: ActionType,
        description: str,
        parameters: Optional[Dict[str, Any]] = None,
        requires_confirmation: bool = False,
    ) -> ReasoningAction:
        """Record the decided action"""
        action = ReasoningAction(
            action_type=action_type,
            description=description,
            parameters=parameters or {},
            requires_confirmation=requires_confirmation,
        )

        # Add as a step
        step = ReasoningStep(
            step_type=ReasoningStepType.ACTION,
            content=f"{action_type.value}: {description}",
            metadata={"action": action},
        )
        self.steps.append(step)
        logger.debug(f"ReAct Action: {action_type.value} - {description}")

        return action

    def complete(
        self,
        response: str,
        confidence_score: float,
        clarifying_questions: Optional[List[str]] = None,
    ) -> ReasoningResult:
        """Complete the reasoning process and return result"""

        # Determine final status
        if clarifying_questions:
            self.status = ReasoningStatus.NEEDS_CLARIFICATION
        elif confidence_score < self.confidence_threshold:
            self.status = ReasoningStatus.NEEDS_CLARIFICATION
        else:
            self.status = ReasoningStatus.COMPLETED

        # Get the last action if any
        final_action = None
        for step in reversed(self.steps):
            if step.step_type == ReasoningStepType.ACTION:
                final_action = step.metadata.get("action")
                break

        result = ReasoningResult(
            status=self.status,
            steps=self.steps.copy(),
            final_action=final_action,
            response=response,
            clarifying_questions=clarifying_questions,
            confidence_score=confidence_score,
            reasoning_visible=self.show_reasoning,
            metadata={
                "total_steps": len(self.steps),
                "max_steps": self.max_steps,
                "threshold": self.confidence_threshold,
            }
        )

        logger.info(
            f"ReAct completed: status={self.status.value}, "
            f"confidence={confidence_score:.2f}, steps={len(self.steps)}"
        )

        return result

    def should_continue(self) -> bool:
        """Check if reasoning should continue"""
        return (
            self.status == ReasoningStatus.IN_PROGRESS
            and len(self.steps) < self.max_steps
        )

    @property
    def step_count(self) -> int:
        """Get current step count"""
        return len(self.steps)

    def get_reasoning_summary(self) -> str:
        """Get a summary of the reasoning process"""
        thoughts = [s.content for s in self.steps if s.step_type == ReasoningStepType.THOUGHT]
        reflections = [s.content for s in self.steps if s.step_type == ReasoningStepType.REFLECTION]

        summary_parts = []
        if thoughts:
            summary_parts.append(f"Considered: {'; '.join(thoughts[:3])}")
        if reflections:
            summary_parts.append(f"Concluded: {reflections[-1]}")

        return " | ".join(summary_parts) if summary_parts else "Direct response"


# =============================================================================
# ReAct Prompt Builder
# =============================================================================

class ReActPromptBuilder:
    """
    Builds prompts that encourage ReAct-style reasoning in the LLM.
    """

    REACT_INSTRUCTION = """
When processing this request, follow this reasoning pattern:

1. **THOUGHT**: What is the user asking? What do I need to understand?
2. **THOUGHT**: Do I have enough information to answer accurately?
3. **THOUGHT**: What are the possible interpretations or approaches?
4. **REFLECTION**: Am I confident I can answer without making assumptions?
5. **ACTION**: Either respond with the answer OR ask clarifying questions

If you're uncertain about ANY aspect:
- List what you're uncertain about
- Ask specific clarifying questions
- Do NOT guess or make up information

Format your internal reasoning like this (I'll parse it):
<reasoning>
THOUGHT: [your thinking here]
THOUGHT: [more thinking if needed]
REFLECTION: [your self-assessment]
ACTION: [respond/clarify]
</reasoning>

Then provide your response to the user.
"""

    @classmethod
    def enhance_prompt(cls, base_prompt: str) -> str:
        """Add ReAct instructions to a prompt"""
        return f"{base_prompt}\n\n{cls.REACT_INSTRUCTION}"

    @classmethod
    def parse_reasoning(cls, response: str) -> tuple[List[ReasoningStep], str]:
        """
        Parse reasoning from LLM response.

        Returns:
            Tuple of (reasoning_steps, clean_response)
        """
        steps = []
        clean_response = response

        # Try to extract reasoning block
        import re
        reasoning_match = re.search(
            r'<reasoning>(.*?)</reasoning>',
            response,
            re.DOTALL | re.IGNORECASE
        )

        if reasoning_match:
            reasoning_text = reasoning_match.group(1)

            # Parse individual steps
            for line in reasoning_text.strip().split('\n'):
                line = line.strip()
                if line.startswith('THOUGHT:'):
                    content = line[8:].strip()
                    steps.append(ReasoningStep(
                        step_type=ReasoningStepType.THOUGHT,
                        content=content,
                    ))
                elif line.startswith('REFLECTION:'):
                    content = line[11:].strip()
                    steps.append(ReasoningStep(
                        step_type=ReasoningStepType.REFLECTION,
                        content=content,
                    ))
                elif line.startswith('ACTION:'):
                    content = line[7:].strip()
                    steps.append(ReasoningStep(
                        step_type=ReasoningStepType.ACTION,
                        content=content,
                    ))
                elif line.startswith('OBSERVATION:'):
                    content = line[12:].strip()
                    steps.append(ReasoningStep(
                        step_type=ReasoningStepType.OBSERVATION,
                        content=content,
                    ))

            # Remove reasoning block from response
            clean_response = re.sub(
                r'<reasoning>.*?</reasoning>\s*',
                '',
                response,
                flags=re.DOTALL | re.IGNORECASE
            ).strip()

        return steps, clean_response
