"""
Core Package - 核心基础设施

Provides foundational utilities used across the project:
- Identifiers: UUID formatting
- Formatting: Display and color utilities
- Paths: Project-relative path utilities

Most common imports:
    from core.identifiers import shorten_id, IDType
    from core.formatting import colorize, format_node_header
    from core.paths import get_project_root, data_path, model_path
"""

# Identifier utilities
from .identifiers import (
    shorten_id,
    IDType,
)

# Formatting utilities
from .formatting import (
    colorize,
    colorize_uuid,
    colorize_status,
    format_node_header,
)

# Path utilities
from .paths import (
    get_project_root,
    get_data_root,
    data_path,
    model_path,
)

__all__ = [
    # Identifiers
    'shorten_id',
    'IDType',
    
    # Formatting
    'colorize',
    'colorize_uuid',
    'colorize_status',
    'format_node_header',
    
    # Paths
    'get_project_root',
    'get_data_root',
    'data_path',
    'model_path',
]
