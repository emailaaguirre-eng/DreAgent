"""Model client with stub fallback."""
import os
from dotenv import load_dotenv

load_dotenv()


def call_model(instruction: str, current_content: str, timeout: int = 20) -> str:
    """
    Call LLM to generate new content.
    
    Model must output one of:
      ADVICE <text>
      PROPOSE_EDIT { path, new_text }
      RUN_COMMAND { cmd }
    
    Falls back to stub if no API key.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        # Stub: simple TODO→DONE replacement
        return current_content.replace("TODO", "DONE")
    
    # Real model integration (placeholder)
    try:
        # TODO: Call Anthropic API with timeout
        # import anthropic
        # client = anthropic.Anthropic(api_key=api_key)
        # response = client.messages.create(...)
        # return response.content[0].text[:50*1024]  # Cap at 50KB
        return current_content.replace("TODO", "DONE")
    except Exception:
        return current_content.replace("TODO", "DONE")
