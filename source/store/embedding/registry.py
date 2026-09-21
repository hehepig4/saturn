"""
Embedding Model Registry

Centralized management of embedding models from configuration.
"""

import json
import threading
from loguru import logger
from pathlib import Path
from typing import Dict, Any, Optional

class EmbeddingModelRegistry:
    """Registry for managing embedding models from configuration."""
    
    _instance = None
    _config = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not EmbeddingModelRegistry._initialized:
            with EmbeddingModelRegistry._lock:
                if not EmbeddingModelRegistry._initialized:
                    self._load_config()
                    EmbeddingModelRegistry._initialized = True
    
    def _load_config(self):
        """Load embedding models configuration."""
        # Config is in source/config/, we are in source/store/embedding/
        config_path = Path(__file__).parent.parent.parent / "config" / "embedding_models.json"
        
        if not config_path.exists():
            logger.warning(f"Embedding models config not found: {config_path}")
            EmbeddingModelRegistry._config = {"models": {}, "default_models": {}, "model_aliases": {}}
            return
        
        with open(config_path, 'r', encoding='utf-8') as f:
            EmbeddingModelRegistry._config = json.load(f)
        
        logger.debug(f"✓ Loaded {len(EmbeddingModelRegistry._config['models'])} embedding models from config")
    
    def get_model_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_id: Model ID or alias
            
        Returns:
            Model configuration dict or None if not found
        """
        # Check if it's an alias
        if model_id in self._config.get("model_aliases", {}):
            model_id = self._config["model_aliases"][model_id]
        
        config = self._config.get("models", {}).get(model_id)
        if config and "local_path" in config:
            p = Path(config["local_path"])
            if not p.is_absolute():
                from core.paths import get_project_root
                config = dict(config)
                config["local_path"] = str(get_project_root() / p)
        return config
    
    def get_default_model(self, modality: str) -> Optional[str]:
        """
        Get default model ID for a modality.
        
        Args:
            modality: Modality name (IMAGE, TEXT, AUDIO, VIDEO)
            
        Returns:
            Model ID or None if no default
        """
        return self._config.get("default_models", {}).get(modality)

# Global singleton instance
_registry = EmbeddingModelRegistry()

def get_registry() -> EmbeddingModelRegistry:
    """Get the global embedding model registry."""
    return _registry

def get_model_config(model_id: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific model."""
    return _registry.get_model_config(model_id)
