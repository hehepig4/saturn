"""
LanceDB Store Package - v3.0

Organized, modular store with clean architecture.

Structure:
    - core/: Base table manager
    - ontology/: Ontology storage

Main Entry Point:
    StoreManager - Unified interface for all operations

Usage:
    from store import StoreManager
    
    store = StoreManager("data/lake/lancedb")
    
    # Query with SQL
    results = store.query(
        "SELECT * FROM nodes WHERE modality = 'IMAGE'",
        purpose="image_search"
    )
"""

from .store_manager import StoreManager
from .store_singleton import get_store
from .core import BaseTableManager

__all__ = [
    # Main interface
    "StoreManager",
    "get_store",
    
    # Core components
    "BaseTableManager",
]

