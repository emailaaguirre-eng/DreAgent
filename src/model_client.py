"""Model client with support for multiple AI providers."""
import os
from dotenv import load_dotenv
import httpx
from typing import Optional
from openai import OpenAI
import anthropic

# Load API keys from the env file in the project root
load_dotenv("drekeys.env", override=True)

# Create clients (ignore system proxy env for stability)
openai_client = OpenAI(http_client=httpx.Client(trust_env=False))
anthropic_client = anthropic.Anthropic(http_client=httpx.Client(trust_env=False))

def ask_llm(prompt: str, provider: str = "openai") -> str:
    if provider == "openai":
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.2,
        )
        return r.choices[0].message.content.strip()

    if provider == "anthropic":
        r = anthropic_client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=256,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
        )
        # r.content[0] has .text on newer SDKs
        return getattr(r.content[0], "text", str(r.content[0])).strip()

    raise ValueError("provider must be 'openai' or 'anthropic'")

def call_model(
    instruction: str, 
    current_content: str, 
    model: Optional[str] = None,
    timeout: int = 30
) -> str:
    """
    Call LLM to generate new content.
    
    Supports:
    - claude (Anthropic Claude)
    - openai (ChatGPT)
    - gemini (Google Gemini)
    - ollama (Local models via Ollama)
    
    Args:
        instruction: What to do with the code
        current_content: Current file content
        model: 'claude', 'openai', 'gemini', 'ollama', or None (uses DEFAULT_MODEL)
        timeout: Max seconds for API call
    
    Returns:
        New file content (capped at 50KB)
    """
    # Determine which model to use
    if model is None:
        model = os.getenv("DEFAULT_MODEL", "claude").lower()
    
    print(f"🤖 Using model: {model}")
    
    # Route to appropriate model
    if model == "claude":
        return _try_with_fallback(_call_claude, instruction, current_content, timeout)
    elif model == "openai":
        return _try_with_fallback(_call_openai, instruction, current_content, timeout)
    elif model == "gemini":
        return _try_with_fallback(_call_gemini, instruction, current_content, timeout)
    elif model == "ollama":
        return _try_with_fallback(_call_ollama, instruction, current_content, timeout)
    else:
        print(f"⚠️  Unknown model '{model}', using stub")
        return _stub_model(current_content)


def _try_with_fallback(model_func, instruction: str, current_content: str, timeout: int) -> str:
    """Try a model, fallback to others if it fails."""
    result = model_func(instruction, current_content, timeout)
    if result:
        return result
    
    # Try fallback models in order
    fallbacks = [_call_claude, _call_openai, _call_gemini, _call_ollama]
    fallbacks.remove(model_func)  # Don't retry the same model
    
    for fallback in fallbacks:
        print(f"⚠️  Trying fallback: {fallback.__name__}")
        result = fallback(instruction, current_content, timeout)
        if result:
            return result
    
    # All models failed, use stub
    print("❌ All models failed, using stub")
    return _stub_model(current_content)


# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

def _call_claude(instruction: str, current_content: str, timeout: int) -> Optional[str]:
    """Call Anthropic Claude API."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found")
        return None
    
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        
        prompt = _build_prompt(instruction, current_content)
        
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        new_content = response.content[0].text
        return _cap_output(new_content)
    
    except Exception as e:
        print(f"❌ Claude API error: {e}")
        return None


def _call_openai(instruction: str, current_content: str, timeout: int) -> Optional[str]:
    """Call OpenAI ChatGPT API."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OPENAI_API_KEY found")
        return None
    
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key, timeout=timeout)
        
        prompt = _build_prompt(instruction, current_content)
        
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-turbo", "gpt-3.5-turbo"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000
        )
        
        new_content = response.choices[0].message.content
        return _cap_output(new_content)
    
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return None


def _call_gemini(instruction: str, current_content: str, timeout: int) -> Optional[str]:
    """Call Google Gemini API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  No GOOGLE_API_KEY found")
        return None
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro")
        
        prompt = _build_prompt(instruction, current_content)
        
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8000,
            )
        )
        
        new_content = response.text
        return _cap_output(new_content)
    
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return None


def _call_ollama(instruction: str, current_content: str, timeout: int) -> Optional[str]:
    """Call local Ollama API."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "codellama")
    
    try:
        import requests
        
        prompt = _build_prompt(instruction, current_content)
        
        response = requests.post(
            f"{ollama_host}/api/generate",
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False
            },
            timeout=timeout
        )
        
        if response.status_code != 200:
            print(f"❌ Ollama error: {response.status_code}")
            return None
        
        new_content = response.json().get("response", "")
        return _cap_output(new_content)
    
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return None


# ============================================================================
# UTILITIES
# ============================================================================

def _build_prompt(instruction: str, current_content: str) -> str:
    """Build consistent prompt for all models."""
    return f"""You are a code editor. The user wants to modify a file.

Current file content:
```
{current_content}
```

Instruction: {instruction}

CRITICAL: Output ONLY the complete new file content. Do not include:
- Explanations
- Markdown code block markers (```)
- Commentary
- Anything except the raw file content

Just output the exact file content that should replace the current content."""


def _cap_output(content: str) -> str:
    """Cap output at 50KB and clean it."""
    # Remove common markdown artifacts
    content = content.strip()
    if content.startswith("```"):
        # Remove first line
        lines = content.split("\n")
        content = "\n".join(lines[1:])
    if content.endswith("```"):
        # Remove last line
        lines = content.split("\n")
        content = "\n".join(lines[:-1])
    
    # Cap at 50KB
    if len(content.encode()) > 50 * 1024:
        content = content[:50 * 1024]
    
    return content.strip()


def _stub_model(current_content: str) -> str:
    """Fallback stub model (TODO→DONE replacement)."""
    print("🤖 Using stub model (TODO→DONE)")
    return current_content.replace("TODO", "DONE")


# ============================================================================
# MODEL INFO
# ============================================================================

def list_available_models() -> dict:
    """List which models are available based on API keys."""
    return {
        "claude": bool(os.getenv("ANTHROPIC_API_KEY")),
        "openai": bool(os.getenv("OPENAI_API_KEY")),
        "gemini": bool(os.getenv("GOOGLE_API_KEY")),
        "ollama": True,  # Always available if Ollama is running
        "stub": True  # Always available
    }


def get_recommended_model() -> str:
    """Get the best available model."""
    available = list_available_models()
    
    # Preference order
    for model in ["claude", "openai", "gemini", "ollama"]:
        if available.get(model):
            return model
    
    return "stub"
