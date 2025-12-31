"""
Model Registry - Single Source of Truth for OpenAI Models

This module:
1. Discovers available models from OpenAI API
2. Selects the best model for each capability (hybrid cost optimization)
3. Caches selections to JSON
4. Provides capability-based model selection for cost optimization
5. Provides get_models_for_mode() function for mode-based model selection
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from openai import OpenAI

# Cache settings
CACHE_FILE = Path(__file__).parent / "model_cache.json"
CACHE_DURATION_HOURS = 24  # Refresh cache every 24 hours

# Capability definitions - Hybrid cost optimization system
# Simple tasks use cheaper models, complex tasks use powerful models
CAPABILITIES = {
    "chat_fast": {
        "description": "Quick/cheap responses - speed and cost optimized (for simple tasks)",
        "preferences": [
            "gpt-5-mini",       # Fast GPT-5 series
            "gpt-5-nano",       # Fastest GPT-5 series
            "gpt-5.1-mini",     # GPT-5.1 mini variant
            "gpt-4o-mini",      # Fastest capable model
            "gpt-3.5-turbo",    # Fast and cheap
        ]
    },
    "chat_deep": {
        "description": "Heavier thinking / longer answers - maximum capability (for complex tasks)",
        "preferences": [
            "gpt-5.1-chat",     # Latest GPT-5.1 chat variant
            "gpt-5.1",          # Latest and greatest
            "gpt-5-chat",       # GPT-5 chat variant
            "gpt-5",            # Previous generation
            "gpt-4-turbo",      # Fallback
            "gpt-4",            # Legacy fallback
        ]
    },
    "chat_default": {
        "description": "Everyday chat - balanced performance and cost (vision-capable)",
        "preferences": [
            "gpt-5.1-chat",     # Latest GPT-5.1 chat variant
            "gpt-5.1",          # Latest GPT-5.1
            "gpt-5-chat",       # GPT-5 chat variant
            "gpt-5",            # GPT-5 base
            "gpt-4o",           # Best general purpose with vision
            "gpt-4-turbo",      # Previous generation with vision
            "gpt-4o-mini",      # Fast and efficient with vision
            "gpt-3.5-turbo",    # Fallback (no vision)
        ]
    },
}

# Fallback models if API call fails
FALLBACK_MODELS = {
    "chat_fast": "gpt-4o-mini",
    "chat_deep": "gpt-4-turbo",
    "chat_default": "gpt-4o",
}

# Mode to capability mapping - Hybrid cost optimization
# Simple tasks (triage, executive assistant) → chat_fast (cheaper)
# Complex tasks (legal, finance, research) → chat_deep (more expensive but powerful)
MODE_TO_CAPABILITY = {
    "General Assistant & Triage": "chat_fast",      # Primary: gpt-5-mini (fast & cheap for everyday triage)
    "IT Support": "chat_deep",                       # Primary: gpt-5.1 (flagship for coding/agentic workflows)
    "Executive Assistant & Operations": "chat_fast", # Primary: gpt-5-mini (speed for summaries, rewriting, scheduling)
    "EIAGUS": "chat_deep",                          # Primary: gpt-5.1 (Grant agent - citation-backed research)
    "Research & Learning": "chat_deep",               # Primary: gpt-5.1 (deep explanations + long-context reasoning)
    "Legal Research & Drafting": "chat_deep",        # Primary: gpt-5.1 (with higher reasoning effort for legal reasoning)
    "Finance & Tax": "chat_deep",                     # Primary: gpt-5.1 (numbers + rules + careful wording for precision)
}

# Mode to model defaults (primary, backup) - Direct model mapping per mode (fallback)
# Using actual OpenAI model names (not renamed)
MODE_MODEL_DEFAULTS = {
    "general_assistant_triage":     ("gpt-5-mini", "gpt-5.1"),
    "it_support_tech_support":      ("gpt-5.1",    "gpt-5-mini"),
    "executive_assistant_ops":      ("gpt-5-mini", "gpt-5.1"),
    "eiagus":                       ("gpt-5.1",    "gpt-5-mini"),  # Grant agent
    "research_learning":            ("gpt-5.1",    "gpt-5-mini"),
    "legal_research_assistant":     ("gpt-5.1",    "gpt-5"),
    "accounting_finance_taxes":     ("gpt-5.1",    "gpt-5-mini"),
}

# Mapping from display mode names to snake_case keys
MODE_NAME_TO_KEY = {
    "General Assistant & Triage": "general_assistant_triage",
    "IT Support": "it_support_tech_support",
    "Executive Assistant & Operations": "executive_assistant_ops",
    "EIAGUS": "eiagus",
    "Research & Learning": "research_learning",
    "Legal Research & Drafting": "legal_research_assistant",
    "Finance & Tax": "accounting_finance_taxes",
}


class ModelRegistry:
    """Manages model discovery and capability-based selection (hybrid cost optimization)"""
    
    def __init__(self, api_key: Optional[str] = None, cache_file: Path = CACHE_FILE):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.cache_file = cache_file
        self.client = None
        self.available_models = []
        self.model_selections = {}  # Cache of capability -> model_id mappings
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
        Get the best model for a given capability (hybrid cost optimization)
        
        Args:
            capability: One of: chat_fast, chat_deep, chat_default
            refresh: If True, force refresh from API
        
        Returns:
            Model ID string (e.g., "gpt-5-mini" for chat_fast, "gpt-5.1" for chat_deep)
        """
        # Refresh if needed or if we don't have selections yet
        if refresh or not self.model_selections:
            self.refresh(force=refresh)
        
        # Return selected model or fallback
        return self.model_selections.get(capability, FALLBACK_MODELS.get(capability, "gpt-4o"))
    
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
            # Check for partial match
            for model in available:
                if (model.startswith(preferred + "-") or model.startswith(preferred + "_")) and model not in fallbacks:
                    fallbacks.append(model)
        
        # Add ultimate fallback if not already included
        ultimate_fallback = FALLBACK_MODELS.get(capability, "gpt-4o")
        if ultimate_fallback not in fallbacks and ultimate_fallback not in self.failed_models:
            fallbacks.append(ultimate_fallback)
        
        return fallbacks
    
    def get_all_models(self) -> List[str]:
        """Get list of all available models"""
        if not self.available_models:
            self.refresh()
        return self.available_models.copy()


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_model_registry(api_key: Optional[str] = None) -> ModelRegistry:
    """Get or create the global model registry instance"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry(api_key=api_key)
    return _registry


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


def get_model_for_capability(capability: str, refresh: bool = False) -> str:
    """
    Convenience function to get model for capability (hybrid cost optimization)
    
    Args:
        capability: One of: chat_fast, chat_deep, chat_default
        refresh: If True, force refresh from API
    
    Returns:
        Model ID string
    """
    registry = get_model_registry()
    return registry.get_model_for_capability(capability, refresh=refresh)


def get_models_for_mode(mode_name: str) -> Tuple[str, str]:
    """
    Get primary and backup models for a given mode name
    Uses hybrid cost optimization: capability-based selection for primary, direct mapping for backup
    
    Args:
        mode_name: Display name of the mode (e.g., "General Assistant & Triage")
    
    Returns:
        Tuple of (primary_model, backup_model) - using actual OpenAI model names
    """
    registry = get_model_registry()
    
    # First, try capability-based selection (hybrid cost optimization)
    capability = MODE_TO_CAPABILITY.get(mode_name)
    if capability:
        try:
            primary_model = registry.get_model_for_capability(capability)
            # Get backup from direct mapping
            mode_key = MODE_NAME_TO_KEY.get(mode_name)
            if mode_key:
                _, backup_model = MODE_MODEL_DEFAULTS.get(mode_key, ("gpt-4o", "gpt-4o-mini"))
                return (primary_model, backup_model)
            return (primary_model, "gpt-4o-mini")
        except Exception as e:
            logging.warning(f"Error getting model for capability {capability}: {e}")
    
    # Fallback to direct model mapping
    mode_key = MODE_NAME_TO_KEY.get(mode_name)
    if not mode_key:
        # Try to match by partial name
        mode_name_lower = mode_name.lower().replace(" ", "_").replace("/", "_").replace("&", "").replace("-", "_")
        for key in MODE_NAME_TO_KEY.values():
            if key in mode_name_lower or mode_name_lower in key:
                mode_key = key
                break
    
    # Get models from defaults
    if mode_key:
        models = MODE_MODEL_DEFAULTS.get(mode_key)
        if models:
            return models
    
    # Ultimate fallback
    return ("gpt-4o", "gpt-4o-mini")


# Example usage
if __name__ == "__main__":
    # Test the registry
    logging.basicConfig(level=logging.INFO)
    
    registry = ModelRegistry()
    
    print("="*70)
    print("MODEL REGISTRY - HYBRID COST OPTIMIZATION SYSTEM")
    print("="*70)
    print("\nFetching models from OpenAI API...")
    selections = registry.refresh(force=True)
    all_models = registry.get_all_models()
    
    print(f"\n✅ Found {len(all_models)} available models")
    print("\n" + "="*70)
    print("CAPABILITY-BASED MODEL SELECTIONS (Hybrid Cost Optimization)")
    print("="*70)
    
    for capability, model_id in selections.items():
        desc = CAPABILITIES[capability]["description"]
        print(f"\n{capability:20} → {model_id:20}")
        print(f"  {desc}")
    
    print("\n" + "="*70)
    print("MODE-TO-CAPABILITY MAPPING")
    print("="*70)
    for mode, capability in MODE_TO_CAPABILITY.items():
        model_id = registry.get_model_for_capability(capability)
        print(f"{mode:40} → {capability:15} → {model_id}")
    
    print("\n" + "="*70)
    print("SAMPLE AVAILABLE MODELS (First 20)")
    print("="*70)
    for model in all_models[:20]:
        print(f"  {model}")
    if len(all_models) > 20:
        print(f"  ... and {len(all_models) - 20} more")

