"""
Path Setup for Standalone Script Execution

This module provides a standardized way to set up Python path for scripts
that need to be run directly (not as part of a package import).

Usage:
    # At the top of any standalone script:
    from _path_setup import setup_source_path
    setup_source_path()
    
    # Then import normally:
    from store.store_manager import LakeStoreManager
    
Note: This is only needed for scripts run directly with `python script.py`.
Regular imports within the package do not need this.
"""

import sys
from pathlib import Path


def setup_source_path() -> Path:
    """
    Add the source directory to Python path if not already present.
    
    This enables standalone scripts to import from the source package
    using absolute imports (e.g., `from store.xxx import yyy`).
    
    Returns:
        Path: The source directory path that was added
        
    Example:
        >>> from _path_setup import setup_source_path
        >>> source_dir = setup_source_path()
        >>> from store.store_manager import LakeStoreManager  # Now works
    """
    source_dir = Path(__file__).parent.resolve()
    source_str = str(source_dir)
    
    if source_str not in sys.path:
        sys.path.insert(0, source_str)
    
    return source_dir


# Auto-setup on import for convenience
# Scripts can just do: import _path_setup
_source_dir = setup_source_path()
