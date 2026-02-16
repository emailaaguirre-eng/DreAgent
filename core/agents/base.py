"""
=============================================================================
HUMMINGBIRD-LEA - Base Agent
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Base class for all AI agents. Defines the common interface and
core functionality that Lea, Chiquis, and Grant share.

Phase 2 Updates:
- Integrated ReAct reasoning loop
- Advanced ambiguity detection
- Multi-factor confidence scoring
- Clarifying questions engine
- Reasoning transparency
=============================================================================
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, AsyncGenerator
from enum import Enum

from core.providers.ollama import Message, OllamaResponse, get_ollama_client, ModelType
from core.providers.router import get_router, RoutingDecision
from core.utils.config import get_settings
from core.providers import get_chat_provider


# Phase 2: Import reasoning components
from core.reasoning import (
    ReasoningEngine,
    ReActLoop,
    ReActPromptBuilder,
    ReasoningResult,
    ReasoningStatus,
    ActionType,
    AdvancedAmbiguityDetector,
    AmbiguityAnalysis,
    ConfidenceScorer,
    ConfidenceAssessment,
    ConfidenceLevel,
    ClarifyingQuestionsEngine,
    TransparencyLevel,
    format_for_display,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Models
# =============================================================================

class AgentConfidenceLevel(Enum):
    """Agent's confidence in its response (legacy compatibility)"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_new_level(cls, level: ConfidenceLevel) -> "AgentConfidenceLevel":
        """Convert from new ConfidenceLevel to legacy format"""
        if level in [ConfidenceLevel.VERY_LOW, ConfidenceLevel.LOW]:
            return cls.LOW
        elif level == ConfidenceLevel.MEDIUM:
            return cls.MEDIUM
        else:
            return cls.HIGH


@dataclass
class AgentResponse:
    """Response from an agent"""
    content: str
    agent: str
    confidence: AgentConfidenceLevel = AgentConfidenceLevel.HIGH
    confidence_score: float = 1.0  # Phase 2: Numeric confidence score
    reasoning: Optional[str] = None  # Internal reasoning (if shown)
    reasoning_steps: Optional[List[Dict]] = None  # Phase 2: Structured reasoning
    clarifying_questions: Optional[List[str]] = None  # Questions to ask user
    sources: Optional[List[str]] = None  # Citations
    needs_clarification: bool = False  # Phase 2: Whether clarification is needed
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ConversationContext:
    """Context for a conversation"""
    messages: List[Message] = field(default_factory=list)
    user_name: str = "Dre"
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Legacy Ambiguity Detection (kept for backwards compatibility)
# =============================================================================

class AmbiguityDetector:
    """
    Legacy ambiguity detector - now wraps AdvancedAmbiguityDetector.
    Kept for backwards compatibility.
    """

    @classmethod
    def detect(cls, message: str) -> List[str]:
        """
        Detect ambiguities in a message.

        Returns:
            List of clarifying questions to ask
        """
        detector = AdvancedAmbiguityDetector(max_questions=2)
        analysis = detector.analyze(message)
        return analysis.get_top_questions(max_questions=2)


# =============================================================================
# Base Agent Class
# =============================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    All agents (Lea, Chiquis, Grant) inherit from this class.
    It provides:
    - Common system prompt structure
    - Conversation management
    - Phase 2: ReAct reasoning loop
    - Phase 2: Advanced ambiguity detection
    - Phase 2: Multi-factor confidence scoring
    - Response generation
    """

    def __init__(self):
        self.settings = get_settings()
        self.ollama = get_ollama_client()
        self.router = get_router()

        # Agent identity (override in subclasses)
        self.name = "Agent"
        self.role = "AI Assistant"
        self.personality = ""
        self.capabilities = []
        self.model_type = ModelType.CHAT

        # Phase 2: Initialize reasoning engine
        self.reasoning_engine = ReasoningEngine(
            agent_name=self.name,
            max_reasoning_steps=self.settings.max_reasoning_steps,
            confidence_threshold=self.settings.confidence_threshold,
            show_reasoning=False,  # Can be toggled for debugging
            max_questions=2,
        )

        logger.info(f"{self.name} agent initialized with Phase 2 reasoning")

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        The system prompt that defines the agent's behavior.
        Must be implemented by subclasses.
        """
        pass

    def _build_full_system_prompt(self, context: ConversationContext) -> str:
        """
        Build the complete system prompt with context and ReAct instructions.
        """
        # Get current time info
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")

        # Time-appropriate greeting
        hour = now.hour
        if 5 <= hour < 12:
            greeting_time = "morning"
        elif 12 <= hour < 17:
            greeting_time = "afternoon"
        elif 17 <= hour < 22:
            greeting_time = "evening"
        else:
            greeting_time = "night"

        # Core anti-hallucination rules
        accuracy_rules = """
## CRITICAL ACCURACY RULES

You MUST follow these rules to avoid hallucination:

1. **NEVER ASSUME** - If something is unclear, ASK before answering
2. **NEVER INVENT** - Do not make up facts, dates, numbers, names, or events
3. **SAY "I DON'T KNOW"** - If you don't have information, admit it clearly
4. **CITE SOURCES** - When referencing documents or data, cite where it came from
5. **CONFIRM CRITICAL ACTIONS** - Before sending emails, creating files, or any irreversible action, confirm details with the user
6. **CHECK CONFIDENCE** - If you're less than 85% confident, express uncertainty

When you detect ambiguity in a request:
- List the possible interpretations
- Ask clarifying questions
- Do NOT guess and proceed

Example of GOOD behavior:
User: "Send that email"
You: "I want to make sure I send the right email. Which email are you referring to?
     - The draft to John about the project?
     - The follow-up to Lisa?
     - A new email you'd like me to compose?"

Example of BAD behavior (NEVER do this):
User: "Send that email"
You: "Done! I've sent the email." (This is hallucination - you don't know which email!)
"""

        # Phase 2: Add ReAct reasoning instructions (skip for OpenAI - not needed)
        from core.providers import get_chat_provider
        chat_provider = get_chat_provider()
        if chat_provider.__class__.__name__ == "OpenAIProvider":
            react_instructions = ""
        else:
            react_instructions = ReActPromptBuilder.REACT_INSTRUCTION

        # Build the full prompt
        full_prompt = f"""{self.system_prompt}

{accuracy_rules}

## Current Context
- Current Date: {date_str}
- Current Time: {time_str}
- Time of Day: {greeting_time}
- User: {context.user_name}

## Response Guidelines
- Be warm and personable, but prioritize accuracy over friendliness
- If asked about something you're unsure of, acknowledge uncertainty
- Show your reasoning when it helps build trust
- Ask clarifying questions rather than making assumptions

{react_instructions}
"""

        return full_prompt

    async def process(
        self,
        user_message: str,
        context: Optional[ConversationContext] = None,
        stream: bool = False,
        show_reasoning: bool = False,
    ) -> AgentResponse:
        """
        Process a user message and generate a response.

        This is the main entry point for agent interactions.
        Now includes Phase 2 reasoning loop.

        Args:
            user_message: The user's message
            context: Conversation context (history, metadata)
            stream: Whether to stream the response
            show_reasoning: Whether to show reasoning to user

        Returns:
            AgentResponse with the agent's reply
        """
        if context is None:
            context = ConversationContext()

        # Phase 2: Start reasoning session
        self.reasoning_engine.start_reasoning()

        # Phase 2: Analyze request for ambiguity
        context_dicts = [{"content": m.content, "role": m.role} for m in context.messages]
        analysis = self.reasoning_engine.analyze_request(user_message, context_dicts)

        # Log reasoning
        self.reasoning_engine.add_thought(f"Analyzing request: '{user_message[:50]}...'")

        if analysis["needs_clarification"]:
            self.reasoning_engine.add_thought(
                f"Detected ambiguity (score: {analysis['ambiguity_score']:.2f})"
            )
            self.reasoning_engine.add_reflection(
                "Should ask for clarification before proceeding"
            )

            # Get clarifying questions
            clarification = analysis.get("clarification")
            if clarification:
                questions = [q.question for q in clarification.questions[:2]]

                # Format the response with questions
                formatted_response = clarification.format(max_questions=2)

                result = self.reasoning_engine.complete_reasoning(
                    response=formatted_response,
                    clarifying_questions=questions,
                )

                return AgentResponse(
                    content=formatted_response,
                    agent=self.name,
                    confidence=AgentConfidenceLevel.MEDIUM,
                    confidence_score=result.confidence_score,
                    clarifying_questions=questions,
                    needs_clarification=True,
                    reasoning=result.formatted_reasoning if show_reasoning else None,
                    reasoning_steps=[
                        {"type": s.step_type.value, "content": s.content}
                        for s in result.steps
                    ] if show_reasoning else None,
                    metadata={
                        "ambiguity_score": analysis["ambiguity_score"],
                        "reasoning_status": result.status.value,
                    }
                )

        # No critical ambiguity - proceed with response generation
        self.reasoning_engine.add_thought("Request is clear enough to proceed")

        # Route to the best model
        routing = self.router.route(
            message=user_message,
            agent=self.name.lower(),
        )
        logger.info(f"Routing decision: {routing.model} - {routing.reason}")

        self.reasoning_engine.add_thought(f"Using model: {routing.model}")

        # Build messages
        system_prompt = self._build_full_system_prompt(context)

        messages = [Message(role="system", content=system_prompt)]

        # Add conversation history
        for msg in context.messages[-self.settings.max_conversation_history:]:
            messages.append(msg)

        # Add the new user message
        messages.append(Message(role="user", content=user_message))

        # Handle streaming
        if stream:
            return AgentResponse(
                content="",  # Will be filled by stream
                agent=self.name,
                confidence=AgentConfidenceLevel.HIGH,
                confidence_score=1.0,
                metadata={"streaming": True, "model": routing.model}
            )

        # Non-streaming response
        chat_provider = get_chat_provider()

        if chat_provider.__class__.__name__ == "OpenAIProvider":
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]
            openai_model = getattr(self.settings, "openai_model_chat", None) or "gpt-4o-mini"
            response = await chat_provider.chat(openai_messages, model=openai_model)
        else:
            response = await self.ollama.chat(
                messages=messages,
                model=routing.model,
                model_type=routing.model_type,
            )

        # Phase 2: Parse reasoning from response (if model included it)
        reasoning_steps, clean_response = ReActPromptBuilder.parse_reasoning(response.content)

        # Add parsed reasoning to our loop
        for step in reasoning_steps:
            if step.step_type.value == "thought":
                self.reasoning_engine.add_thought(step.content)
            elif step.step_type.value == "reflection":
                self.reasoning_engine.add_reflection(step.content)

        # Phase 2: Assess confidence with new multi-factor scorer
        confidence_assessment = self.reasoning_engine.assess_response(
            response=clean_response,
            question=user_message,
            has_sources=bool(self._extract_sources(clean_response)),
        )

        self.reasoning_engine.add_reflection(
            f"Confidence assessment: {confidence_assessment.level.value} ({confidence_assessment.percentage}%)"
        )

        # Complete reasoning
        result = self.reasoning_engine.complete_reasoning(
            response=clean_response,
            clarifying_questions=None,
        )

        # Prepare final response
        final_content = clean_response

        # Add uncertainty note if confidence is low
        if confidence_assessment.should_express_uncertainty:
            uncertainty_phrases = confidence_assessment.uncertainty_phrases
            if uncertainty_phrases:
                # The response already might have uncertainty - don't double up
                if not any(phrase.lower() in clean_response.lower() for phrase in uncertainty_phrases):
                    # Add a subtle confidence note
                    final_content = format_for_display(
                        clean_response,
                        confidence=confidence_assessment,
                        level=TransparencyLevel.MINIMAL,
                    )

        return AgentResponse(
            content=final_content,
            agent=self.name,
            confidence=AgentConfidenceLevel.from_new_level(confidence_assessment.level),
            confidence_score=confidence_assessment.raw_score,
            reasoning=result.formatted_reasoning if show_reasoning else None,
            reasoning_steps=[
                {"type": s.step_type.value, "content": s.content}
                for s in result.steps
            ] if show_reasoning else None,
            sources=self._extract_sources(clean_response),
            metadata={
                "model": (getattr(self.settings, "openai_model_chat", None) or routing.model),
                "complexity": routing.complexity.name,
                "eval_count": response.eval_count,
                "confidence_factors": [
                    {"factor": f.factor.value, "score": f.score}
                    for f in confidence_assessment.factors
                ],
                "reasoning_status": result.status.value,
                "reasoning_steps_count": len(result.steps),
            }
        )

    async def process_stream(
        self,
        user_message: str,
        context: Optional[ConversationContext] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Process a message and stream the response.

        Usage:
            async for chunk in agent.process_stream("Hello!"):
                print(chunk, end="", flush=True)
        """
        if context is None:
            context = ConversationContext()

        # Phase 2: Quick ambiguity check before streaming
        context_dicts = [{"content": m.content, "role": m.role} for m in context.messages]
        analysis = self.reasoning_engine.analyze_request(user_message, context_dicts)

        if analysis["needs_clarification"]:
            # If we need clarification, yield the clarification request
            clarification = analysis.get("clarification")
            if clarification:
                yield clarification.format(max_questions=2)
                return

        # Route to the best model
        routing = self.router.route(
            message=user_message,
            agent=self.name.lower(),
        )

        # Build messages
        system_prompt = self._build_full_system_prompt(context)

        messages = [Message(role="system", content=system_prompt)]

        for msg in context.messages[-self.settings.max_conversation_history:]:
            messages.append(msg)

        messages.append(Message(role="user", content=user_message))

     # Stream the response
        # Stream the response
        # Check provider - use OpenAI if configured
        chat_provider = get_chat_provider()

        if chat_provider.__class__.__name__ == "OpenAIProvider":
            openai_messages = [{"role": m.role, "content": m.content} for m in messages]
            openai_model = getattr(self.settings, "openai_model_chat", None) or "gpt-4o-mini"
            response = await chat_provider.chat(openai_messages, model=openai_model)
            yield response.content
        else:
            async for chunk in self.ollama.chat_stream(
                messages=messages,
                model=routing.model,
                model_type=routing.model_type,
            ):
                yield chunk

    def _extract_sources(self, response: str) -> Optional[List[str]]:
        """Extract sources/citations from a response"""
        import re

        sources = []

        # Look for common citation patterns
        # Pattern: [1], [2], etc.
        numbered_refs = re.findall(r'\[(\d+)\]', response)
        if numbered_refs:
            sources.extend([f"Reference [{n}]" for n in set(numbered_refs)])

        # Pattern: (Source: ...)
        source_matches = re.findall(r'\(Source:\s*([^)]+)\)', response, re.IGNORECASE)
        sources.extend(source_matches)

        # Pattern: According to ...
        according_matches = re.findall(r'[Aa]ccording to ([^,\.]+)', response)
        sources.extend(according_matches[:3])  # Limit to 3

        return sources if sources else None

    def _assess_confidence(self, response: str) -> AgentConfidenceLevel:
        """
        Legacy confidence assessment method.
        Now wraps the new ConfidenceScorer for backwards compatibility.
        """
        scorer = ConfidenceScorer()
        assessment = scorer.assess_response(response)
        return AgentConfidenceLevel.from_new_level(assessment.level)

    def add_message_to_context(
        self,
        context: ConversationContext,
        role: str,
        content: str,
    ) -> ConversationContext:
        """Add a message to the conversation context"""
        context.messages.append(Message(role=role, content=content))
        return context

    def set_reasoning_visibility(self, show: bool):
        """Toggle reasoning visibility for debugging"""
        self.reasoning_engine.show_reasoning = show

    def get_reasoning_summary(self) -> str:
        """Get a summary of the last reasoning session"""
        return self.reasoning_engine.react_loop.get_reasoning_summary()
