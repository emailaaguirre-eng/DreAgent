"""
=============================================================================
HUMMINGBIRD-LEA - Smart Router
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Intelligent routing that selects the best model based on task complexity.
Optimized for local Ollama with optional cloud fallback.
=============================================================================
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from core.utils.config import get_settings
from .ollama import OllamaClient, ModelType, get_ollama_client

logger = logging.getLogger(__name__)


# =============================================================================
# Task Classification
# =============================================================================

class TaskComplexity(Enum):
    """Complexity levels for routing decisions"""
    SIMPLE = 1      # Greetings, simple questions
    MODERATE = 2    # General assistance, summaries
    COMPLEX = 3     # Analysis, research, multi-step reasoning
    CODE = 4        # Code generation and debugging
    VISION = 5      # Image analysis


@dataclass
class RoutingDecision:
    """Result of routing decision"""
    model: str
    model_type: ModelType
    reason: str
    complexity: TaskComplexity


# =============================================================================
# Smart Router
# =============================================================================

class SmartRouter:
    """
    Routes requests to the optimal model based on task analysis.
    
    This keeps costs at $0 by using local Ollama models intelligently.
    Different models excel at different tasks:
    - llama3.1:8b → General chat, reasoning, analysis
    - deepseek-coder → Code generation, debugging
    - llava → Image understanding
    
    Usage:
        router = SmartRouter()
        decision = router.route("Write a Python function to sort a list")
        print(f"Using {decision.model} because: {decision.reason}")
    """
    
    # Keywords for task classification
    SIMPLE_KEYWORDS = [
        "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
        "how are you", "what's up", "good morning", "good afternoon",
        "good evening", "what time", "what day", "who are you", "your name",
    ]
    
    CODE_KEYWORDS = [
        "code", "function", "class", "debug", "error", "bug", "script",
        "python", "javascript", "html", "css", "sql", "api", "programming",
        "variable", "loop", "array", "list", "dictionary", "import",
        "def ", "async", "await", "return", "print(", "console.log",
        "fix this", "write a program", "implement", "refactor",
    ]
    
    COMPLEX_KEYWORDS = [
        "analyze", "analysis", "research", "compare", "contrast",
        "evaluate", "synthesize", "summarize", "explain in detail",
        "pros and cons", "implications", "strategy", "recommend",
        "legal", "case law", "statute", "court", "precedent",
        "incentive", "tax credit", "site selection", "economic",
        "why does", "how does", "what are the reasons",
    ]
    
    VISION_KEYWORDS = [
        "image", "picture", "photo", "screenshot", "diagram",
        "look at", "see this", "what's in", "describe this image",
        "analyze this image", "ocr", "read this",
    ]
    
    def __init__(self):
        """Initialize the router"""
        self.settings = get_settings()
        self.ollama = get_ollama_client()
        logger.info("SmartRouter initialized")
    
    def _classify_complexity(self, message: str, agent: str = "lea") -> TaskComplexity:
        """
        Classify the complexity of a user message.
        
        Args:
            message: User's message
            agent: Current agent (lea, chiquis, grant)
        
        Returns:
            TaskComplexity enum
        """
        message_lower = message.lower().strip()
        
        # Check for vision tasks first (requires special model)
        if any(kw in message_lower for kw in self.VISION_KEYWORDS):
            return TaskComplexity.VISION
        
        # Check for code tasks (Chiquis specialty)
        if agent == "chiquis" or any(kw in message_lower for kw in self.CODE_KEYWORDS):
            return TaskComplexity.CODE
        
        # Check for complex analysis (Grant specialty)
        if agent == "grant" or any(kw in message_lower for kw in self.COMPLEX_KEYWORDS):
            return TaskComplexity.COMPLEX
        
        # Check for simple greetings
        if any(kw in message_lower for kw in self.SIMPLE_KEYWORDS):
            return TaskComplexity.SIMPLE
        
        # Short messages are usually simple
        if len(message_lower) < 30:
            return TaskComplexity.SIMPLE
        
        # Default to moderate
        return TaskComplexity.MODERATE
    
    def route(
        self,
        message: str,
        agent: str = "lea",
        has_image: bool = False,
        force_model: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route a request to the optimal model.
        
        Args:
            message: User's message
            agent: Current agent (lea, chiquis, grant)
            has_image: Whether the request includes an image
            force_model: Override automatic selection
        
        Returns:
            RoutingDecision with model and reasoning
        """
        # If model is forced, use it
        if force_model:
            return RoutingDecision(
                model=force_model,
                model_type=ModelType.CHAT,
                reason="Model explicitly specified",
                complexity=TaskComplexity.MODERATE,
            )
        
        # If image is present, use vision model
        if has_image:
            return RoutingDecision(
                model=self.settings.ollama_model_vision,
                model_type=ModelType.VISION,
                reason="Image analysis requires vision model",
                complexity=TaskComplexity.VISION,
            )
        
        # Classify the task
        complexity = self._classify_complexity(message, agent)
        
        # Route based on complexity
        if complexity == TaskComplexity.CODE:
            return RoutingDecision(
                model=self.settings.ollama_model_code,
                model_type=ModelType.CODE,
                reason="Code task routed to coding model (deepseek-coder)",
                complexity=complexity,
            )
        
        if complexity == TaskComplexity.VISION:
            return RoutingDecision(
                model=self.settings.ollama_model_vision,
                model_type=ModelType.VISION,
                reason="Vision task routed to llava",
                complexity=complexity,
            )
        
        # All other tasks use the main chat model
        # llama3.1:8b handles simple, moderate, and complex well
        reason_map = {
            TaskComplexity.SIMPLE: "Simple greeting/question",
            TaskComplexity.MODERATE: "General assistance task",
            TaskComplexity.COMPLEX: "Complex analysis task",
        }
        
        return RoutingDecision(
            model=self.settings.ollama_model_chat,
            model_type=ModelType.CHAT,
            reason=reason_map.get(complexity, "General task"),
            complexity=complexity,
        )
    
    def get_model_for_agent(self, agent: str) -> str:
        """
        Get the default model for a specific agent.
        
        Args:
            agent: Agent name (lea, chiquis, grant)
        
        Returns:
            Model name string
        """
        if agent == "chiquis":
            return self.settings.ollama_model_code
        # Lea and Grant both use the chat model
        return self.settings.ollama_model_chat


# =============================================================================
# Singleton Instance
# =============================================================================

_router: Optional[SmartRouter] = None


def get_router() -> SmartRouter:
    """Get or create the router singleton"""
    global _router
    if _router is None:
        _router = SmartRouter()
    return _router
