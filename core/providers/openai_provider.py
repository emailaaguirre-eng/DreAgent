from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from core.utils.config import get_settings


@dataclass
class OpenAIChatResult:
    content: str
    raw: Dict[str, Any]
    eval_count: int = 0


class OpenAIProvider:
    """
    OpenAI chat provider using Chat Completions API.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: int = 120) -> None:
        self.settings = get_settings()
        self.api_key = api_key or getattr(self.settings, "openai_api_key", None)
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        self.base_url = "https://api.openai.com/v1"
        self.timeout = timeout

    async def chat(self, messages: List[Dict[str, str]], model: str) -> OpenAIChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.85,
            "max_tokens": 4096,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            r.raise_for_status()
            data = r.json()

        content = data["choices"][0]["message"]["content"]

        return OpenAIChatResult(content=content.strip(), raw=data)


_openai_client: Optional[OpenAIProvider] = None


def get_openai_client() -> OpenAIProvider:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIProvider()
    return _openai_client
