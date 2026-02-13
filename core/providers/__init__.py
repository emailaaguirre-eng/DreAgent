"""
AI Providers for Hummingbird-LEA
"""

from .ollama import (
    OllamaClient,
    OllamaResponse,
    EmbeddingResponse,
    Message,
    ModelType,
    get_ollama_client,
    ollama,
)

from .router import (
    SmartRouter,
    RoutingDecision,
    TaskComplexity,
    get_router,
)

__all__ = [
    "OllamaClient",
    "OllamaResponse",
    "EmbeddingResponse",
    "Message",
    "ModelType",
    "get_ollama_client",
    "ollama",
    "SmartRouter",
    "RoutingDecision",
    "TaskComplexity",
    "get_router",
]
