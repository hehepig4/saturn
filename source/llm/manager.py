"""
LLM Manager

Singleton pattern for efficient LLM initialization and caching.
Supports purpose-based model selection (decision, toolcall, default).

Note: LLM configuration is read from config/llm_models.json.
      Use get_llm_by_purpose() for purpose-based model selection.
"""

from typing import Optional, List, Dict, Any, Literal
from pathlib import Path
import os
import json
import threading
from loguru import logger
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from config.truncation_limits import TruncationLimits


# Default LLM configuration (fallback when llm_models.json is not available)
# These values are rarely used - prefer get_llm_by_purpose() with llm_models.json
_DEFAULT_PROVIDER = 'openai'
_DEFAULT_MODEL = 'model'
_DEFAULT_TEMPERATURE = 0
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
_DEFAULT_BASE_URL = None

# Type for model purpose
ModelPurpose = Literal["decision", "toolcall", "default", "local"]

# Global config cache with thread lock
_config_cache: Optional[Dict[str, Any]] = None
_config_lock = threading.Lock()


def load_llm_config() -> Dict[str, Any]:
    """Load LLM model configuration from JSON (thread-safe)."""
    global _config_cache
    
    # Fast path: already loaded
    if _config_cache is not None:
        return _config_cache
    
    # Thread-safe loading
    with _config_lock:
        # Double-check after acquiring lock
        if _config_cache is not None:
            return _config_cache
        
        config_path = Path(__file__).parent.parent / "config" / "llm_models.json"
        
        try:
            with open(config_path, 'r') as f:
                _config_cache = json.load(f)
            logger.debug(f"Loaded LLM model config from {config_path}")
            return _config_cache
        except Exception as e:
            logger.warning(f"Failed to load LLM config: {e}, using defaults")
            _config_cache = {
                "models": {},
                "purpose_mapping": {
                    "decision": "default_model",
                    "toolcall": "default_model", 
                    "default": "default_model"
                },
                "api_config": {
                    "base_url": _DEFAULT_BASE_URL,
                    "max_retries": _DEFAULT_MAX_RETRIES
                }
            }
            return _config_cache


class LLMManager:
    """
    Singleton LLM manager to cache and reuse LLM instances.
    
    Supports purpose-based model selection:
    - decision: For agent decision making
    - toolcall: For tool calling operations
    - default: General purpose
    
    This avoids the performance overhead of repeatedly initializing
    the same model with the same configuration.
    """
    
    _instance: Optional['LLMManager'] = None
    _llm_cache: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_llm_by_purpose(
        self,
        purpose: ModelPurpose = "default",
        tools: Optional[List] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> BaseChatModel:
        """
        Get LLM instance based on purpose (decision, toolcall, or default).
        
        Args:
            purpose: Model purpose ('decision', 'toolcall', 'default')
            tools: Optional list of tools to bind to LLM
            override_config: Optional config overrides (provider, model_name, etc.)
        
        Returns:
            Cached or newly created LLM instance
        
        Example:
            >>> # Get decision-making model
            >>> llm = manager.get_llm_by_purpose("decision")
            >>> 
            >>> # Get tool-calling model with tools bound
            >>> llm = manager.get_llm_by_purpose("toolcall", tools=[my_tool])
            >>>
            >>> # Get local vLLM model
            >>> llm = manager.get_llm_by_purpose("local")
        """
        # Load config
        config = load_llm_config()
        
        # Get model key for this purpose
        model_key = config.get("purpose_mapping", {}).get(purpose, "default_model")
        model_config = config.get("models", {}).get(model_key, {})
        
        if not model_config:
            logger.warning(f"No model config found for purpose '{purpose}', using defaults")
            model_config = {
                "provider": _DEFAULT_PROVIDER,
                "model_name": _DEFAULT_MODEL,
                "temperature": _DEFAULT_TEMPERATURE
            }
        
        # Apply overrides if provided
        if override_config:
            model_config = {**model_config, **override_config}
        
        # Extract config values
        model_provider = model_config.get("provider", _DEFAULT_PROVIDER)
        model_name = model_config.get("model_name", _DEFAULT_MODEL)
        temperature = model_config.get("temperature", _DEFAULT_TEMPERATURE)
        max_tokens = model_config.get("max_tokens")  # Extract max_tokens from merged config
        frequency_penalty = model_config.get("frequency_penalty")
        presence_penalty = model_config.get("presence_penalty")
        seed = model_config.get("seed", 42)  # Default seed for reproducibility
        
        # Check for model-specific API config override (e.g., local vLLM server)
        api_config_override = model_config.get("api_config_override", {})
        api_config = config.get("api_config", {})
        
        # Model-specific override takes precedence
        base_url = api_config_override.get("base_url") or api_config.get("base_url", _DEFAULT_BASE_URL)
        max_retries = api_config_override.get("max_retries") or api_config.get("max_retries", _DEFAULT_MAX_RETRIES)
        
        # API key: model-specific override > api_config > global default
        api_key = api_config_override.get("api_key") or api_config.get("api_key") or _DEFAULT_API_KEY
        if not api_key and ("localhost" in base_url or "127.0.0.1" in base_url):
            api_key = "EMPTY"  # vLLM doesn't require real API key
        
        # Use standard get_llm with extracted config
        return self.get_llm(
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
            tools=tools,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )
    
    def get_llm(
        self,
        model_provider: str = _DEFAULT_PROVIDER,
        model_name: str = _DEFAULT_MODEL,
        temperature: float = _DEFAULT_TEMPERATURE,
        max_tokens: Optional[int] = None,
        max_retries: Optional[int] = None,
        api_key: str = _DEFAULT_API_KEY,
        base_url: str = _DEFAULT_BASE_URL,
        tools: Optional[List] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = 42
    ) -> BaseChatModel:
        """
        Get or create cached LLM instance.
        
        Args:
            model_provider: Provider name (e.g., 'openai', 'anthropic')
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response (None for provider default)
            api_key: API key for authentication
            base_url: Base URL for API endpoint
            tools: Optional list of tools to bind to LLM
            frequency_penalty: Reduce repetition of exact tokens (-2.0 to 2.0)
            presence_penalty: Encourage new topics (-2.0 to 2.0)
            seed: Random seed for reproducibility (default: 42, None to disable)
        
        Returns:
            Cached or newly created LLM instance with tools bound
        """
        # Create cache key from configuration
        cache_key = (
            model_provider,
            model_name,
            temperature,
            max_tokens,
            base_url,
            frequency_penalty,
            presence_penalty,
            seed,
            tuple(tool.name if hasattr(tool, 'name') else str(tool) for tool in (tools or []))
        )
        
        # Return cached instance if available
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]
        
        # Create new LLM instance with all parameters
        # Pass frequency_penalty and presence_penalty via kwargs
        # Use configured timeout to prevent hanging on slow/failed requests
        llm = init_chat_model(
            model_provider=model_provider,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=TruncationLimits.LLM_TIMEOUT,
            max_retries=max_retries or _DEFAULT_MAX_RETRIES,
            api_key=api_key,
            base_url=base_url,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed
        )
        
        # Bind tools if provided
        if tools:
            llm = llm.bind_tools(tools)
        
        # Cache and return
        self._llm_cache[cache_key] = llm
        return llm
    
    def clear_cache(self):
        """Clear all cached LLM instances."""
        self._llm_cache.clear()


# Global instance
_llm_manager = LLMManager()


def get_llm_by_purpose(
    purpose: ModelPurpose = "default",
    tools: Optional[List] = None,
    override_config: Optional[Dict[str, Any]] = None,
    temperature_override: Optional[float] = None
) -> BaseChatModel:
    """
    Get LLM instance based on purpose.
    
    Args:
        purpose: Model purpose ('decision', 'toolcall', 'default')
        tools: Optional list of tools to bind
        override_config: Optional config overrides
        temperature_override: Override temperature (for retry with different temp)
    
    Returns:
        LLM instance configured for the specified purpose
    
    Example:
        >>> # Get decision model
        >>> decision_llm = get_llm_by_purpose("decision")
        >>> 
        >>> # Get toolcall model with tools
        >>> from tools.node_tools import node_search
        >>> toolcall_llm = get_llm_by_purpose("toolcall", tools=[node_search])
        
        >>> # For retry with different temperature
        >>> llm = get_llm_by_purpose("default", temperature_override=0.1)
    """
    # Apply temperature override to config
    if temperature_override is not None:
        if override_config is None:
            override_config = {}
        override_config = dict(override_config)  # Make a copy
        override_config["temperature"] = temperature_override
    
    return _llm_manager.get_llm_by_purpose(
        purpose=purpose,
        tools=tools,
        override_config=override_config
    )


def get_llm(
    model_provider: str = _DEFAULT_PROVIDER,
    model_name: str = _DEFAULT_MODEL,
    temperature: float = _DEFAULT_TEMPERATURE,
    api_key: str = _DEFAULT_API_KEY,
    base_url: str = _DEFAULT_BASE_URL,
    tools: Optional[List] = None
) -> BaseChatModel:
    """
    Convenience function to get cached LLM instance.
    
    Returns:
        Cached or newly created LLM instance
    """
    return _llm_manager.get_llm(
        model_provider=model_provider,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        tools=tools
    )


def clear_llm_cache():
    """Clear all cached LLM instances."""
    _llm_manager.clear_cache()


__all__ = [
    'LLMManager',
    'ModelPurpose',
    'load_llm_config',
    'get_llm',
    'get_llm_by_purpose',
    'clear_llm_cache',
]
