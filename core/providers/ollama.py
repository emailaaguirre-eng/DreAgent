"""
=============================================================================
HUMMINGBIRD-LEA - Ollama Provider
Powered by CoDre-X | B & D Servicing LLC
=============================================================================
Async client for Ollama local AI.
Handles chat, code generation, vision, and embeddings.
=============================================================================
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import httpx

from core.utils.config import get_settings

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Models
# =============================================================================

class ModelType(Enum):
    """Types of models available"""
    CHAT = "chat"
    CODE = "code"
    VISION = "vision"
    EMBED = "embed"


@dataclass
class Message:
    """A chat message"""
    role: str  # "system", "user", or "assistant"
    content: str
    images: Optional[List[str]] = None  # Base64 encoded images for vision


@dataclass
class OllamaResponse:
    """Response from Ollama"""
    content: str
    model: str
    done: bool
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None  # tokens generated


@dataclass
class EmbeddingResponse:
    """Embedding response from Ollama"""
    embedding: List[float]
    model: str


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """
    Async client for Ollama API.
    
    Usage:
        client = OllamaClient()
        
        # Check if Ollama is available
        if await client.is_available():
            # Chat
            response = await client.chat([
                Message(role="user", content="Hello!")
            ])
            print(response.content)
            
            # Stream
            async for chunk in client.chat_stream(messages):
                print(chunk, end="", flush=True)
    """
    
    def __init__(self, host: Optional[str] = None, timeout: int = 120):
        """
        Initialize the Ollama client.
        
        Args:
            host: Ollama server URL (default: from settings)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.host = host or settings.ollama_host
        self.timeout = timeout
        
        # Model mapping
        self._models = {
            ModelType.CHAT: settings.ollama_model_chat,
            ModelType.CODE: settings.ollama_model_code,
            ModelType.VISION: settings.ollama_model_vision,
            ModelType.EMBED: settings.ollama_model_embed,
        }
        
        logger.info(f"OllamaClient initialized with host: {self.host}")
    
    def get_model(self, model_type: ModelType) -> str:
        """Get the model name for a given type"""
        return self._models.get(model_type, self._models[ModelType.CHAT])
    
    async def is_available(self) -> bool:
        """Check if Ollama server is available"""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.host}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on the Ollama server"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.host}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [m["name"] for m in data.get("models", [])]
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.
        This can take a while for large models.
        """
        try:
            async with httpx.AsyncClient(timeout=3600) as client:  # 1 hour timeout
                response = await client.post(
                    f"{self.host}/api/pull",
                    json={"name": model},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model}: {e}")
            return False
    
    async def chat(
        self,
        messages: List[Message],
        model_type: ModelType = ModelType.CHAT,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> OllamaResponse:
        """
        Send a chat request to Ollama (non-streaming).
        
        Args:
            messages: List of Message objects
            model_type: Type of model to use
            model: Override model name (optional)
            temperature: Creativity (0-1)
            max_tokens: Max tokens to generate
        
        Returns:
            OllamaResponse with the generated content
        """
        model_name = model or self.get_model(model_type)
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.images:
                m["images"] = msg.images
            ollama_messages.append(m)
        
        payload = {
            "model": model_name,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.host}/api/chat",
                    json=payload,
                )
                
                if response.status_code != 200:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    return OllamaResponse(
                        content=f"Error: Ollama returned status {response.status_code}",
                        model=model_name,
                        done=True
                    )
                
                data = response.json()
                
                return OllamaResponse(
                    content=data.get("message", {}).get("content", ""),
                    model=model_name,
                    done=data.get("done", True),
                    total_duration=data.get("total_duration"),
                    eval_count=data.get("eval_count"),
                )
                
        except httpx.TimeoutException:
            logger.error(f"Ollama request timed out after {self.timeout}s")
            return OllamaResponse(
                content="Error: Request timed out. The model may be loading or the query is too complex.",
                model=model_name,
                done=True
            )
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            return OllamaResponse(
                content=f"Error: {str(e)}",
                model=model_name,
                done=True
            )
    
    async def chat_stream(
        self,
        messages: List[Message],
        model_type: ModelType = ModelType.CHAT,
        model: Optional[str] = None,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response from Ollama.
        
        Usage:
            async for chunk in client.chat_stream(messages):
                print(chunk, end="", flush=True)
        """
        model_name = model or self.get_model(model_type)
        
        # Convert messages to Ollama format
        ollama_messages = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.images:
                m["images"] = msg.images
            ollama_messages.append(m)
        
        payload = {
            "model": model_name,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.host}/api/chat",
                    json=payload,
                ) as response:
                    if response.status_code != 200:
                        yield f"Error: Ollama returned status {response.status_code}"
                        return
                    
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                import json
                                data = json.loads(line)
                                content = data.get("message", {}).get("content", "")
                                if content:
                                    yield content
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"Error: {str(e)}"
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> Optional[EmbeddingResponse]:
        """
        Generate embedding vector for text.
        
        Args:
            text: Text to embed
            model: Model to use (default: embedding model from settings)
        
        Returns:
            EmbeddingResponse with the embedding vector
        """
        model_name = model or self.get_model(ModelType.EMBED)
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.host}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": text,
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Embedding error: {response.status_code}")
                    return None
                
                data = response.json()
                
                return EmbeddingResponse(
                    embedding=data.get("embedding", []),
                    model=model_name,
                )
                
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None
    
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str = "Describe this image in detail.",
        model: Optional[str] = None,
    ) -> OllamaResponse:
        """
        Analyze an image using the vision model.
        
        Args:
            image_base64: Base64 encoded image
            prompt: Question or instruction about the image
            model: Vision model to use
        
        Returns:
            OllamaResponse with the analysis
        """
        model_name = model or self.get_model(ModelType.VISION)
        
        messages = [
            Message(
                role="user",
                content=prompt,
                images=[image_base64]
            )
        ]
        
        return await self.chat(
            messages=messages,
            model=model_name,
            model_type=ModelType.VISION,
        )


# =============================================================================
# Singleton Instance
# =============================================================================

_ollama_client: Optional[OllamaClient] = None


def get_ollama_client() -> OllamaClient:
    """Get or create the Ollama client singleton"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client


# Convenience export
ollama = get_ollama_client()
