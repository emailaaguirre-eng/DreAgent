"""AI Providers for Hummingbird-LEA"""
from core.utils.config import get_settings

from .ollama import (
    OllamaClient,
    OllamaResponse,
    EmbeddingResponse,
    Message,
    ModelType,
    get_ollama_client,
    ollama,
)

from .openai_provider import OpenAIProvider, get_openai_client

from .router import (
    SmartRouter,
    RoutingDecision,
    TaskComplexity,
    get_router,
)


def get_chat_provider():
    """
    Use OpenAI for chat if OPENAI_API_KEY is set; otherwise use Ollama.
    """
    s = get_settings()
    if getattr(s, "openai_api_key", None):
        return get_openai_client()
    return get_ollama_client()


__all__ = [
    "OllamaClient",
    "OllamaResponse",
    "EmbeddingResponse",
    "Message",
    "ModelType",
    "get_ollama_client",
    "ollama",
    "OpenAIProvider",
    "get_openai_client",
    "get_chat_provider",
    "SmartRouter",
    "RoutingDecision",
    "TaskComplexity",
    "get_router",
]

