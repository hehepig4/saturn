"""
Embedding Function Registry Manager

This module handles registration of custom embedding functions with LanceDB
based on configuration files. It provides a centralized way to manage and
instantiate embedding functions.
"""

from loguru import logger
from typing import Dict, Optional, Any
from pathlib import Path
import threading

from store.embedding.config import (
    EmbeddingFunctionConfig,
    EMBEDDING_FUNCTIONS,
    get_embedding_function_config,
    get_default_function_for_modality,
)
from store.embedding.custom_embeddings import (
    LocalCLIPTextEmbedding,
    LocalCLIPImageEmbedding,
    LocalBGEM3Embedding,
    EmbeddingFunction,
)

class EmbeddingFunctionRegistry:
    """
    Registry for managing and creating embedding functions.
    
    This class acts as a factory that creates embedding function instances
    based on configuration. It handles caching to avoid recreating the same
    functions multiple times.
    
    Thread-safe: Uses locks to prevent concurrent model loading.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._instances: Dict[str, EmbeddingFunction] = {}
        self._config_cache: Dict[str, EmbeddingFunctionConfig] = {}
        self._lock = threading.Lock()  # Protect against concurrent access
        logger.info("Initialized EmbeddingFunctionRegistry")
    
    def register_function(
        self,
        name: str,
        config: Optional[EmbeddingFunctionConfig] = None,
        force_recreate: bool = False,
    ) -> EmbeddingFunction:
        """
        Register and instantiate an embedding function.
        
        Args:
            name: Name of the embedding function
            config: Optional custom config (uses default if None)
            force_recreate: Force recreation even if cached
            
        Returns:
            Instantiated embedding function
            
        Raises:
            ValueError: If configuration is invalid
            KeyError: If function name not found
        """
        # Check cache first (without lock for performance)
        if name in self._instances and not force_recreate:
            logger.debug(f"Using cached embedding function: {name}")
            return self._instances[name]
        
        # Use lock to prevent concurrent model loading
        with self._lock:
            # Double-check after acquiring lock (another thread may have created it)
            if name in self._instances and not force_recreate:
                logger.debug(f"Using cached embedding function: {name} (from concurrent thread)")
                return self._instances[name]
            
            # Get configuration
            if config is None:
                config = get_embedding_function_config(name)
            
            # Validate configuration
            is_valid, error = config.validate()
            if not is_valid:
                raise ValueError(f"Invalid configuration for '{name}': {error}")
            
            # Create function instance based on type
            logger.info(f"Creating embedding function: {name} ({config.function_type})")
            
            if config.function_type == "text":
                func = self._create_text_function(config)
            elif config.function_type == "image":
                func = self._create_image_function(config)
            else:
                raise ValueError(f"Unsupported function type: {config.function_type}")
            
            # Cache the instance
            self._instances[name] = func
            self._config_cache[name] = config
            
            logger.info(f"✓ Registered embedding function: {name}")
            logger.info(f"    Model: {config.model_name}")
            logger.info(f"    Path: {config.model_path}")
            logger.info(f"    Dimension: {config.embedding_dim}")
            
            return func
    
    def _create_text_function(
        self,
        config: EmbeddingFunctionConfig
    ) -> EmbeddingFunction:
        """Create a text embedding function based on model type."""
        # Check model name to determine which implementation to use
        model_name = config.model_name.lower()
        
        if "bge" in model_name:
            # Use BGE-M3 embedding
            max_length = config.custom_params.get("max_length", 512)
            return LocalBGEM3Embedding(
                model_path=config.model_path,
                device=config.device,
                batch_size=config.batch_size,
                max_length=max_length,
                normalize=config.normalize,
            )
        else:
            # Default to CLIP text encoder
            return LocalCLIPTextEmbedding(
                model_path=config.model_path,
                device=config.device,
                batch_size=config.batch_size,
                normalize=config.normalize,
            )
    
    def _create_image_function(
        self,
        config: EmbeddingFunctionConfig
    ) -> LocalCLIPImageEmbedding:
        """Create an image embedding function."""
        # Currently only supports CLIP image encoder
        # Can be extended to support other models
        return LocalCLIPImageEmbedding(
            model_path=config.model_path,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize,
        )
    
    def get_function(self, name: str) -> Optional[EmbeddingFunction]:
        """
        Get a previously registered function.
        
        Args:
            name: Function name
            
        Returns:
            Embedding function or None if not registered
        """
        return self._instances.get(name)
    
    def get_function_for_modality(
        self,
        modality: str,
        custom_function_name: Optional[str] = None
    ) -> EmbeddingFunction:
        """
        Get or create an embedding function for a modality.
        
        Args:
            modality: Modality type (TEXT, IMAGE, etc.)
            custom_function_name: Optional custom function name
            
        Returns:
            Embedding function instance
        """
        # Use custom function if specified, otherwise use default
        function_name = custom_function_name or get_default_function_for_modality(modality)
        
        # Register if not already registered
        if function_name not in self._instances:
            return self.register_function(function_name)
        
        return self._instances[function_name]
    
    def unregister_function(self, name: str) -> bool:
        """
        Unregister a function and free resources.
        
        Args:
            name: Function name
            
        Returns:
            True if unregistered, False if not found
        """
        if name in self._instances:
            del self._instances[name]
            if name in self._config_cache:
                del self._config_cache[name]
            logger.info(f"Unregistered embedding function: {name}")
            return True
        return False
    
    def clear_all(self):
        """Clear all registered functions."""
        count = len(self._instances)
        self._instances.clear()
        self._config_cache.clear()
        logger.info(f"Cleared {count} embedding function(s)")
    
    def list_registered(self) -> list[str]:
        """
        List all registered function names.
        
        Returns:
            List of function names
        """
        return list(self._instances.keys())
    
    def get_config(self, name: str) -> Optional[EmbeddingFunctionConfig]:
        """
        Get configuration for a registered function.
        
        Args:
            name: Function name
            
        Returns:
            Configuration or None if not found
        """
        return self._config_cache.get(name)
    
    def register_all_default_functions(self) -> Dict[str, EmbeddingFunction]:
        """
        Register all default embedding functions from config.
        
        Returns:
            Dict mapping function name to instance
        """
        logger.info("Registering all default embedding functions...")
        
        registered = {}
        for name in EMBEDDING_FUNCTIONS.keys():
            try:
                func = self.register_function(name)
                registered[name] = func
            except Exception as exc:
                logger.error(f"Failed to register '{name}': {exc}")
        
        logger.info(f"✓ Registered {len(registered)} embedding function(s)")
        return registered

# ============================= GLOBAL REGISTRY INSTANCE =============================

# Global registry instance (singleton pattern)
_global_registry: Optional[EmbeddingFunctionRegistry] = None

def get_registry() -> EmbeddingFunctionRegistry:
    """
    Get the global embedding function registry.
    
    Returns:
        Global registry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = EmbeddingFunctionRegistry()
    return _global_registry

# ============================= CONVENIENCE FUNCTIONS =============================

def get_embedding_function(name: str) -> Optional[EmbeddingFunction]:
    """
    Convenience function to get a registered embedding function.
    
    Args:
        name: Function name
        
    Returns:
        Embedding function or None
    """
    registry = get_registry()
    return registry.get_function(name)

def get_embedding_function_for_modality(
    modality: str,
    custom_function_name: Optional[str] = None
) -> EmbeddingFunction:
    """
    Convenience function to get embedding function for a modality.
    
    Args:
        modality: Modality type
        custom_function_name: Optional custom function name
        
    Returns:
        Embedding function instance
    """
    registry = get_registry()
    return registry.get_function_for_modality(modality, custom_function_name)


