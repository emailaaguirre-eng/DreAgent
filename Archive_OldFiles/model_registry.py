"""
Model Registry - Single Source of Truth for OpenAI Models

This module:
1. Discovers available models from OpenAI API
2. Selects the best model for each capability
3. Caches selections to JSON
4. Provides get_model_for_capability() function
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from openai import OpenAI

# Cache settings
CACHE_FILE = Path(__file__).parent / "model_cache.json"
CACHE_DURATION_HOURS = 24  # Refresh cache every 24 hours

# Capability definitions
CAPABILITIES = {
    "chat_default": {
        "description": "Everyday chat - balanced performance and cost (vision-capable)",
        "preferences": [
            "gpt-5.1-chat",     # Latest GPT-5.1 chat variant
            "gpt-5.1",          # Latest GPT-5.1
            "gpt-5-chat",       # GPT-5 chat variant
            "gpt-5",            # GPT-5 base
            "gpt-4o",           # Best general purpose with vision
            "gpt-4-turbo",      # Previous generation with vision
            "gpt-5-mini",       # Balanced GPT-5 option (if vision-capable)
            "gpt-5-nano",       # Fast GPT-5 option (if vision-capable)
            "gpt-4o-mini",      # Fast and efficient with vision
            "gpt-3.5-turbo",    # Fallback (no vision)
        ]
    },
    "chat_deep": {
        "description": "Heavier thinking / longer answers - maximum capability",
        "preferences": [
            "gpt-5.1-chat",     # Latest GPT-5.1 chat variant
            "gpt-5.1",          # Latest and greatest
            "gpt-5-chat",       # GPT-5 chat variant
            "gpt-5",            # Previous generation
            "gpt-5-mini",       # Fast GPT-5 option
            "gpt-4.1",          # Smartest non-reasoning
            "gpt-4-turbo",      # Fallback
            "gpt-4",            # Legacy fallback
        ]
    },
    "chat_fast": {
        "description": "Quick/cheap responses - speed and cost optimized",
        "preferences": [
            "gpt-5-mini",       # Fast GPT-5 series
            "gpt-5-nano",       # Fastest GPT-5 series
            "gpt-5.1-mini",     # GPT-5.1 mini variant
            "gpt-4o-mini",      # Fastest capable model
            "gpt-3.5-turbo",    # Fast and cheap
            "gpt-4.1-nano",     # If available
        ]
    },
    "reasoning": {
        "description": "Advanced reasoning models (o3/o1 series)",
        "preferences": [
            "o3",               # Latest reasoning model
            "o3-mini",          # Smaller reasoning model
            "o1",               # Previous reasoning model
            "o1-mini",          # Smaller o1
            "gpt-5.1",          # Fallback to GPT-5.1 if reasoning not available
            "gpt-5",            # Fallback to GPT-5
        ]
    },
    "embedding": {
        "description": "Text embeddings",
        "preferences": [
            "text-embedding-3-large",   # Best embeddings
            "text-embedding-3-small",   # Smaller embeddings
            "text-embedding-ada-002",   # Legacy fallback
        ]
    },
    "vision": {
        "description": "Vision/image understanding",
        "preferences": [
            "gpt-5.1-chat",     # Latest GPT-5.1 chat (supports vision)
            "gpt-5.1",          # GPT-5.1 if vision-capable
            "gpt-5-chat",       # GPT-5 chat (supports vision)
            "gpt-5",            # GPT-5 if vision-capable
            "gpt-4o",           # Best vision model
            "gpt-4-turbo",      # Previous generation with vision
            "gpt-4-vision-preview",  # Legacy vision model
        ]
    },
    "code": {
        "description": "Code generation and understanding",
        "preferences": [
            "gpt-5.1-codex",    # Best for coding
            "gpt-5-codex",      # Previous generation codex
            "gpt-5.1-chat",     # GPT-5.1 chat variant
            "gpt-5.1",          # Fallback to GPT-5.1
            "gpt-5-chat",       # GPT-5 chat variant
            "gpt-5",            # Fallback to GPT-5
            "gpt-4o",           # General purpose fallback
        ]
    },
}

# Fallback models if API call fails
FALLBACK_MODELS = {
    "chat_default": "gpt-4o",
    "chat_deep": "gpt-4-turbo",
    "chat_fast": "gpt-4o-mini",
    "reasoning": "gpt-4-turbo",  # No reasoning model fallback
    "embedding": "text-embedding-3-small",
    "vision": "gpt-4o",
    "code": "gpt-4o",
}

# Mode to model defaults (primary, backup) - Direct model mapping per mode
MODE_MODEL_DEFAULTS = {
    "general_assistant_triage":     ("gpt-5-mini", "gpt-5.1"),
    "it_support_tech_support":      ("gpt-5.1",    "gpt-5-mini"),
    "executive_assistant_ops":      ("gpt-5-mini", "gpt-5.1"),
    "incentives":                   ("gpt-5.1",    "gpt-5-mini"),
    "research_learning":            ("gpt-5.1",    "gpt-5-mini"),
    "legal_research_assistant":     ("gpt-5.1",    "gpt-5"),
    "accounting_finance_taxes":     ("gpt-5.1",    "gpt-5-mini"),
}

# Mapping from display mode names to snake_case keys
MODE_NAME_TO_KEY = {
    "General Assistant & Triage": "general_assistant_triage",
    "IT Support": "it_support_tech_support",
    "Executive Assistant & Operations": "executive_assistant_ops",
    "Incentives": "incentives",
    "Research & Learning": "research_learning",
    "Legal Research Assistant": "legal_research_assistant",
    "Accounting/Finance/Taxes": "accounting_finance_taxes",
}


class ModelRegistry:
    """Manages model discovery and capability-based selection"""
    
    def __init__(self, api_key: Optional[str] = None, cache_file: Path = CACHE_FILE):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cache_file = cache_file
        self.client = None
        self.available_models = []
        self.model_selections = {}
        self.failed_models = set()  # Track models that have failed
        
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                logging.warning(f"Could not initialize OpenAI client: {e}")
    
    def _load_cache(self) -> Optional[Dict]:
        """Load cached model selections"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid
            cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
            age = datetime.now() - cache_time
            
            if age < timedelta(hours=CACHE_DURATION_HOURS):
                return cache_data
            else:
                logging.info(f"Cache expired (age: {age})")
                return None
        except Exception as e:
            logging.warning(f"Error loading cache: {e}")
            return None
    
    def _save_cache(self, model_selections: Dict, available_models: List[str]):
        """Save model selections to cache"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'model_selections': model_selections,
                'available_models': available_models,
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logging.info(f"Saved model cache to {self.cache_file}")
        except Exception as e:
            logging.warning(f"Error saving cache: {e}")
    
    def _fetch_available_models(self) -> List[str]:
        """Fetch available models from OpenAI API"""
        # Try using requests library first (more reliable for model discovery)
        try:
            import requests
            if self.api_key:
                logging.info("Fetching models using requests library...")
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.get(
                    "https://api.openai.com/v1/models",
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    available = [model.get("id", "") for model in data.get("data", []) if model.get("id")]
                    logging.info(f"API returned {len(available)} total models via requests")
                    if available:
                        # Filter to only chat/completion models (exclude deprecated and non-chat models)
                        chat_models = [
                            m for m in available 
                            if not m.startswith('dall-e') 
                            and not m.startswith('whisper')
                            and not m.startswith('tts-')
                            and not m.startswith('moderation')
                            and not m.startswith('text-embedding')  # Embeddings are separate
                            and 'deprecated' not in m.lower()
                            and m not in ['babbage', 'davinci', 'curie', 'ada']  # Legacy completion models
                        ]
                        
                        logging.info(f"Filtered to {len(chat_models)} chat/completion models")
                        if chat_models:
                            logging.debug(f"Sample models: {chat_models[:10]}")
                        
                        # Check for GPT-5 models specifically
                        gpt5_models = [m for m in chat_models if 'gpt-5' in m.lower()]
                        if gpt5_models:
                            logging.info(f"✅ GPT-5 models found: {gpt5_models}")
                        else:
                            logging.warning("⚠️ No GPT-5 models found in API response. This may be due to:")
                            logging.warning("   - API key usage tier (GPT-5 requires tiers 1-5)")
                            logging.warning("   - Organization verification status")
                            logging.warning("   - Models not yet available for your account")
                            logging.warning("   See: https://help.openai.com/en/articles/10362446")
                        
                        return sorted(chat_models)
                else:
                    logging.warning(f"OpenAI API returned status {response.status_code}")
        except ImportError:
            logging.debug("requests library not available, using OpenAI client")
        except Exception as e:
            logging.debug(f"Requests method failed: {e}, trying OpenAI client")
        
        # Fallback to OpenAI client
        if not self.client:
            logging.warning("No OpenAI client available, using fallback models")
            return []
        
        try:
            logging.info("Calling OpenAI API to fetch available models...")
            models = self.client.models.list()
            
            if not models or not hasattr(models, 'data'):
                logging.warning("API returned no models or invalid response")
                return []
            
            available = [model.id for model in models.data]
            logging.info(f"API returned {len(available)} total models via OpenAI client")
            
            # Filter to only chat/completion models (exclude deprecated and non-chat models)
            chat_models = [
                m for m in available 
                if not m.startswith('dall-e') 
                and not m.startswith('whisper')
                and not m.startswith('tts-')
                and not m.startswith('moderation')
                and not m.startswith('text-embedding')  # Embeddings are separate
                and 'deprecated' not in m.lower()
                and m not in ['babbage', 'davinci', 'curie', 'ada']  # Legacy completion models
            ]
            
            logging.info(f"Filtered to {len(chat_models)} chat/completion models")
            if chat_models:
                logging.debug(f"Sample models: {chat_models[:10]}")
            
            # Check for GPT-5 models specifically
            gpt5_models = [m for m in chat_models if 'gpt-5' in m.lower()]
            if gpt5_models:
                logging.info(f"✅ GPT-5 models found: {gpt5_models}")
            else:
                logging.warning("⚠️ No GPT-5 models found in API response. This may be due to:")
                logging.warning("   - API key usage tier (GPT-5 requires tiers 1-5)")
                logging.warning("   - Organization verification status")
                logging.warning("   - Models not yet available for your account")
                logging.warning("   See: https://help.openai.com/en/articles/10362446")
            
            return sorted(chat_models)
        except Exception as e:
            import traceback
            logging.error(f"Error fetching models from API: {e}")
            logging.debug(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _select_best_model(self, capability: str, available_models: List[str]) -> str:
        """Select the best available model for a capability"""
        if capability not in CAPABILITIES:
            logging.warning(f"Unknown capability: {capability}")
            return FALLBACK_MODELS.get(capability, "gpt-4o")
        
        preferences = CAPABILITIES[capability]["preferences"]
        
        # Normalize model IDs to lowercase for comparison
        available_models_lower = [m.lower() for m in available_models]
        
        # Try each preference in order
        for preferred in preferences:
            preferred_lower = preferred.lower()
            
            # Check for exact match
            if preferred in available_models:
                logging.info(f"Selected {preferred} for {capability}")
                return preferred
            
            # Check for partial match (e.g., "gpt-4o" matches "gpt-4o-2024-08-06")
            for idx, model in enumerate(available_models):
                model_lower = model.lower()
                
                # Enhanced matching for GPT-5 variants
                # Handle: gpt-5.1, gpt-5.1-chat, gpt-5.1-chat-latest, gpt-5-codex, etc.
                matches = (
                    model == preferred or
                    model_lower == preferred_lower or
                    model_lower.startswith(preferred_lower) or
                    model_lower.startswith(preferred_lower + "-") or
                    preferred_lower in model_lower or
                    # Special handling for GPT-5 variants
                    (preferred_lower.startswith("gpt-5") and "gpt-5" in model_lower and
                     (preferred_lower.replace("gpt-5", "") in model_lower.replace("gpt-5", "") or
                      model_lower.startswith("gpt-5")))
                )
                
                if matches:
                    logging.info(f"Selected {model} (matches {preferred}) for {capability}")
                    return model
        
        # If no preference matched, try to find any model that might work
        # For reasoning, look for o1/o3
        if capability == "reasoning":
            for model in available_models:
                if model.startswith("o1") or model.startswith("o3"):
                    logging.info(f"Selected {model} for {capability}")
                    return model
        
        # For embedding, look for embedding models
        if capability == "embedding":
            for model in available_models:
                if "embedding" in model.lower():
                    logging.info(f"Selected {model} for {capability}")
                    return model
        
        # For vision, look for vision-capable models
        if capability == "vision":
            for model in available_models:
                if "vision" in model.lower() or model.startswith("gpt-4"):
                    logging.info(f"Selected {model} for {capability}")
                    return model
        
        # Fallback to default
        fallback = FALLBACK_MODELS.get(capability, "gpt-4o")
        logging.warning(f"No suitable model found for {capability}, using fallback: {fallback}")
        return fallback
    
    def refresh(self, force: bool = False) -> Dict[str, str]:
        """Refresh model selections from API"""
        # Try to load from cache first (unless forced)
        if not force:
            cache = self._load_cache()
            if cache:
                self.model_selections = cache.get('model_selections', {})
                self.available_models = cache.get('available_models', [])
                logging.info("Loaded model selections from cache")
                return self.model_selections
        
        # Fetch fresh data from API
        logging.info("Fetching models from OpenAI API...")
        self.available_models = self._fetch_available_models()
        
        # If API call failed, try to use cached data even if expired
        if not self.available_models:
            cache = self._load_cache()
            if cache:
                self.model_selections = cache.get('model_selections', {})
                self.available_models = cache.get('available_models', [])
                logging.warning(f"API call failed, using expired cache with {len(self.available_models)} models")
                if self.available_models:
                    logging.info(f"Cached models: {self.available_models[:10]}...")
                return self.model_selections
            else:
                logging.error("API call failed and no cache available!")
                return {}
        
        # Select best model for each capability
        self.model_selections = {}
        for capability in CAPABILITIES.keys():
            self.model_selections[capability] = self._select_best_model(
                capability, 
                self.available_models
            )
        
        # Save to cache
        self._save_cache(self.model_selections, self.available_models)
        
        return self.model_selections
    
    def get_model_for_capability(self, capability: str, refresh: bool = False) -> str:
        """
        Get the best model for a given capability
        
        Args:
            capability: One of: chat_default, chat_deep, chat_fast, reasoning, embedding, vision, code
            refresh: If True, force refresh from API
        
        Returns:
            Model ID string (e.g., "gpt-4o")
        """
        # Refresh if needed or if we don't have selections yet
        if refresh or not self.model_selections:
            self.refresh(force=refresh)
        
        # Return selected model or fallback
        return self.model_selections.get(capability, FALLBACK_MODELS.get(capability, "gpt-4o"))
    
    def get_all_models(self) -> List[str]:
        """Get list of all available models"""
        if not self.available_models:
            self.refresh()
        return self.available_models.copy()
    
    def get_selections(self) -> Dict[str, str]:
        """Get all current model selections"""
        if not self.model_selections:
            self.refresh()
        return self.model_selections.copy()
    
    def mark_model_failed(self, model_id: str, error: str = "", error_type: str = "unknown"):
        """
        Mark a model as failed and automatically recover by selecting alternatives
        
        Returns:
            dict with recovery info: {"recovered": bool, "new_model": str, "capability": str, "reason": str}
        """
        self.failed_models.add(model_id)
        
        recovery_info = {
            "recovered": False,
            "new_model": None,
            "capability": None,
            "reason": error,
            "error_type": error_type
        }
        
        logging.warning(f"🔴 Model {model_id} marked as failed. Error: {error} (Type: {error_type})")
        
        # If this model is currently selected for any capability, automatically recover
        for capability, selected_model in self.model_selections.items():
            if selected_model == model_id:
                logging.info(f"🔄 Auto-recovering capability '{capability}' from failed model {model_id}")
                
                # Re-select model for this capability, excluding failed model
                available = [m for m in self.available_models if m not in self.failed_models]
                new_model = self._select_best_model(capability, available)
                
                if new_model and new_model != model_id:
                    self.model_selections[capability] = new_model
                    recovery_info["recovered"] = True
                    recovery_info["new_model"] = new_model
                    recovery_info["capability"] = capability
                    recovery_info["reason"] = f"Auto-switched to {new_model} for {capability}"
                    logging.info(f"✅ Auto-recovery successful: {capability} now uses {new_model}")
                else:
                    recovery_info["reason"] = f"Could not find alternative for {capability}"
                    logging.error(f"❌ Auto-recovery failed for {capability}")
        
        # Save updated selections
        if self.model_selections:
            self._save_cache(self.model_selections, self.available_models)
        
        return recovery_info
    
    def get_fallback_models(self, capability: str, exclude_model: Optional[str] = None) -> List[str]:
        """
        Get list of fallback models for a capability, excluding failed models and specified model
        
        Returns prioritized list of alternative models
        """
        if capability not in CAPABILITIES:
            return []
        
        preferences = CAPABILITIES[capability]["preferences"]
        available = [m for m in self.available_models if m not in self.failed_models]
        
        if exclude_model:
            available = [m for m in available if m != exclude_model]
        
        fallbacks = []
        for preferred in preferences:
            # Check for exact match
            if preferred in available and preferred not in fallbacks:
                fallbacks.append(preferred)
            # Check for partial match (e.g., "gpt-4o" matches "gpt-4o-2024-08-06")
            for model in available:
                if (model.startswith(preferred + "-") or model.startswith(preferred + "_")) and model not in fallbacks:
                    fallbacks.append(model)
        
        # Add ultimate fallback if not already included
        ultimate_fallback = FALLBACK_MODELS.get(capability, "gpt-4o")
        if ultimate_fallback not in fallbacks and ultimate_fallback not in self.failed_models:
            # Check if any variant of ultimate fallback exists
            ultimate_found = False
            for model in available:
                if model.startswith(ultimate_fallback.split("-")[0]):  # e.g., "gpt-4"
                    if model not in fallbacks:
                        fallbacks.append(model)
                    ultimate_found = True
                    break
            if not ultimate_found and ultimate_fallback not in self.failed_models:
                fallbacks.append(ultimate_fallback)
        
        return fallbacks
    
    def get_recovery_info(self, failed_model: str, capability: Optional[str] = None) -> dict:
        """
        Get detailed recovery information for a failed model
        
        Returns:
            dict with recovery options and explanations
        """
        info = {
            "failed_model": failed_model,
            "is_failed": failed_model in self.failed_models,
            "capability": capability,
            "alternatives": [],
            "recommended": None,
            "explanation": ""
        }
        
        if capability:
            alternatives = self.get_fallback_models(capability, exclude_model=failed_model)
            info["alternatives"] = alternatives
            info["recommended"] = alternatives[0] if alternatives else None
            info["explanation"] = f"For {capability}, {len(alternatives)} alternative(s) available"
        else:
            # Try to find capability that uses this model
            for cap, model in self.model_selections.items():
                if model == failed_model:
                    info["capability"] = cap
                    alternatives = self.get_fallback_models(cap, exclude_model=failed_model)
                    info["alternatives"] = alternatives
                    info["recommended"] = alternatives[0] if alternatives else None
                    info["explanation"] = f"Model used for {cap}, {len(alternatives)} alternative(s) available"
                    break
        
        return info
    
    def is_model_available(self, model_id: str) -> bool:
        """Check if a model is available (not failed and in available list)"""
        return model_id not in self.failed_models and (
            model_id in self.available_models or 
            any(model_id.startswith(m) for m in self.available_models)
        )
    
    def clear_failed_models(self, model_id: Optional[str] = None):
        """
        Clear failed model(s) - useful if models become available again
        
        Args:
            model_id: Specific model to clear, or None to clear all
        """
        if model_id:
            if model_id in self.failed_models:
                self.failed_models.remove(model_id)
                logging.info(f"✅ Cleared failed status for {model_id}")
                return True
        else:
            count = len(self.failed_models)
            self.failed_models.clear()
            logging.info(f"✅ Cleared all {count} failed model(s)")
            return count > 0
        return False
    
    def get_failed_models(self) -> List[str]:
        """Get list of currently failed models"""
        return list(self.failed_models)
    
    def get_health_status(self) -> dict:
        """
        Get overall health status of the model registry
        
        Returns:
            dict with health metrics
        """
        total_models = len(self.available_models)
        failed_count = len(self.failed_models)
        working_count = total_models - failed_count
        
        health_status = {
            "total_models": total_models,
            "working_models": working_count,
            "failed_models": failed_count,
            "health_percentage": (working_count / total_models * 100) if total_models > 0 else 0,
            "failed_list": list(self.failed_models),
            "status": "healthy" if failed_count == 0 else "degraded" if working_count > 0 else "critical"
        }
        
        return health_status


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(api_key: Optional[str] = None) -> ModelRegistry:
    """Get or create the global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(api_key=api_key)
    return _registry


def get_model_for_capability(capability: str, refresh: bool = False) -> str:
    """
    Convenience function to get model for capability
    
    Args:
        capability: One of: chat_default, chat_deep, chat_fast, reasoning, embedding, vision, code
        refresh: If True, force refresh from API
    
    Returns:
        Model ID string
    """
    registry = get_model_registry()
    return registry.get_model_for_capability(capability, refresh=refresh)


def refresh_models(force: bool = False) -> Dict[str, str]:
    """
    Refresh model selections from API
    
    Args:
        force: If True, force refresh even if cache is valid
    
    Returns:
        Dictionary of capability -> model_id
    """
    registry = get_model_registry()
    return registry.refresh(force=force)


def get_models_for_mode(mode_name: str) -> tuple[str, str]:
    """
    Get primary and backup models for a given mode name
    
    Args:
        mode_name: Display name of the mode (e.g., "General Assistant & Triage")
    
    Returns:
        Tuple of (primary_model, backup_model)
    """
    # Convert display name to snake_case key
    mode_key = MODE_NAME_TO_KEY.get(mode_name, mode_name.lower().replace(" ", "_").replace("/", "_").replace("&", "").replace("-", "_"))
    
    # Get models from defaults
    models = MODE_MODEL_DEFAULTS.get(mode_key)
    if models:
        return models
    
    # Fallback: try to find by partial match
    mode_key_lower = mode_key.lower()
    for key, value in MODE_MODEL_DEFAULTS.items():
        if key in mode_key_lower or mode_key_lower in key:
            return value
    
    # Ultimate fallback
    return ("gpt-4o", "gpt-4o-mini")


# Example usage
if __name__ == "__main__":
    # Test the registry
    logging.basicConfig(level=logging.INFO)
    
    registry = ModelRegistry()
    
    print("Available capabilities:")
    for cap in CAPABILITIES.keys():
        model = registry.get_model_for_capability(cap)
        print(f"  {cap}: {model}")
    
    print("\nAll available models:")
    models = registry.get_all_models()
    for model in models[:20]:  # Show first 20
        print(f"  {model}")
    if len(models) > 20:
        print(f"  ... and {len(models) - 20} more")

