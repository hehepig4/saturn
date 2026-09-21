"""
Identifier Helper Functions

Convenience functions for common ID operations.
"""

from typing import Optional


def shorten_id(full_id: str, length: Optional[int] = None, keep_prefix: bool = True) -> str:
    """
    Shorten UUID for display purposes.
    
    Args:
        full_id: Full UUID string (e.g., "tbl_abc123-def456-...")
        length: Display length for the ID portion (default 8)
        keep_prefix: Whether to keep type prefix
        
    Returns:
        Shortened ID string (e.g., "tbl_abc123" or "abc123")
    """
    if not full_id:
        return ""
    
    if length is None:
        length = 8
    
    # Check if there's a prefix (e.g., "tbl_", "col_")
    parts = full_id.split("_", 1)
    
    if len(parts) == 2 and len(parts[0]) <= 4:
        # Has prefix
        prefix = parts[0]
        uuid_part = parts[1]
        
        # Remove hyphens for cleaner display
        uuid_clean = uuid_part.replace("-", "")
        short_uuid = uuid_clean[:length]
        
        if keep_prefix:
            return f"{prefix}_{short_uuid}"
        else:
            return short_uuid
    else:
        # No prefix, just shorten the UUID
        uuid_clean = full_id.replace("-", "")
        return uuid_clean[:length]


__all__ = [
    'shorten_id',
]
