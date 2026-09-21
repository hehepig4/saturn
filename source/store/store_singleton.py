"""
StoreManager Global Singleton

Provides global singleton access to StoreManager for LLM tools.
This ensures consistent database connection across all tool calls.

Usage:
    from store.store_singleton import get_store
    
    @tool
    def my_tool(node_uuid: str):
        store = get_store()
        return store.query(f"uuid = '{node_uuid}'", table="nodes")
    
    # For experiments with isolated DB:
    exp_store = create_store("/path/to/experiment/db")
"""

from typing import Optional
from pathlib import Path
from loguru import logger
from store.store_manager import StoreManager
from core.paths import get_db_path

# Global singleton instance
_store_instance: Optional[StoreManager] = None
_default_db_path: Optional[str] = None


def _get_default_db_path() -> str:
    """Lazily resolve default DB path, respecting SATURN_DB_PATH env var."""
    global _default_db_path
    if _default_db_path is None:
        _default_db_path = str(get_db_path())
    return _default_db_path


def get_store(db_path: Optional[str] = None) -> StoreManager:
    """
    Get or create StoreManager singleton instance.
    
    Args:
        db_path: Database path (optional, uses default if None)
                 If provided on first call, becomes the default path
    
    Returns:
        StoreManager singleton instance
    
    Example:
        >>> store = get_store()
        >>> results = store.query("modality = 'IMAGE'", table="nodes")
    """
    global _store_instance, _default_db_path
    
    # Update default path if provided
    if db_path is not None:
        _default_db_path = db_path
    
    # Create instance if not exists
    if _store_instance is None:
        resolved = _get_default_db_path()
        logger.info(f"Initializing StoreManager singleton at: {resolved}")
        _store_instance = StoreManager(resolved)
        logger.info("StoreManager singleton initialized successfully")
    
    return _store_instance


def create_store(db_path: str) -> StoreManager:
    """
    Create a new StoreManager instance (non-singleton).
    
    Use this for experiments that need isolated database directories.
    The returned instance is independent of the global singleton.
    
    Args:
        db_path: Database path for the new instance
    
    Returns:
        New StoreManager instance
    
    Example:
        >>> exp_store = create_store("/path/to/experiment/db")
        >>> # Use exp_store for experiment-specific operations
    """
    logger.info(f"Creating new StoreManager instance at: {db_path}")
    return StoreManager(db_path)


def reset_store() -> None:
    """
    Reset the global singleton instance.
    
    Use this when switching between databases (e.g., after experiment setup).
    The next call to get_store() will create a new instance.
    """
    global _store_instance, _default_db_path
    if _store_instance is not None:
        logger.info("Resetting StoreManager singleton")
        _store_instance = None
    _default_db_path = None


# Convenience exports
__all__ = [
    'get_store',
    'create_store',
    'reset_store',
]
