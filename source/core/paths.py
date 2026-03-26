"""
Path Utilities

Provides project-relative path resolution to avoid hardcoded paths.
"""

from pathlib import Path
import os


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Priority:
    1. SATURN_ROOT environment variable (highest priority)
    2. Auto-detection by finding directory containing 'source/'
    3. Fallback to parent directory
    
    Returns:
        Path to project root
    """
    # Priority 1: Use environment variable if set
    if 'SATURN_ROOT' in os.environ:
        root = Path(os.environ['SATURN_ROOT'])
        if root.is_dir():
            return root
    
    # Priority 2: Auto-detect by finding 'source' directory
    current = Path(__file__).resolve()
    for parent in [current, *current.parents]:
        if (parent / 'source').is_dir():
            return parent
    
    # Priority 3: Fallback to parent directory
    return Path(__file__).parent.parent.parent


def get_data_root() -> Path:
    """Get data directory (PROJECT_ROOT/data)"""
    root = os.getenv('SATURN_DATA_DIR')
    if root:
        return Path(root)
    return get_project_root() / 'data'


def get_model_root() -> Path:
    """Get model directory (PROJECT_ROOT/model)"""
    root = os.getenv('SATURN_MODEL_DIR')
    if root:
        return Path(root)
    return get_project_root() / 'model'


def get_db_path() -> Path:
    """Get database path"""
    path = os.getenv('SATURN_DB_PATH')
    if path:
        p = Path(path)
        # Resolve relative paths against project root, not cwd
        if not p.is_absolute():
            p = get_project_root() / p
        return p
    return get_data_root() / 'lake' / 'lancedb'


# Convenience path generators
def data_path(*parts: str) -> Path:
    """Get path relative to data directory"""
    return get_data_root().joinpath(*parts)


def model_path(*parts: str) -> Path:
    """Get path relative to model directory"""
    return get_model_root().joinpath(*parts)


def lake_data_path(*parts: str) -> Path:
    """Get path in data/lake/"""
    return data_path('lake', *parts)


def ensure_project_cwd() -> Path:
    """
    Ensure current working directory is the project root.
    
    This prevents relative paths like 'data/cache/...' from being written
    to incorrect locations (e.g., source/data/cache/) when scripts are
    run from subdirectories.
    
    Returns:
        Path: The project root directory
    """
    import os
    project_root = get_project_root()
    if Path.cwd() != project_root:
        os.chdir(project_root)
    return project_root
