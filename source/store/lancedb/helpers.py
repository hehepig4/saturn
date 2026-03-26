"""
LanceDB Helper Functions

Utilities specific to LanceDB operations, including timestamp handling.
"""

from datetime import datetime, timezone


def get_timestamp_ms() -> datetime:
    """
    Get current timestamp with millisecond precision for LanceDB.
    
    LanceDB's timestamp[ms, tz=UTC] schema cannot accept microsecond precision
    from Python's datetime.now(). This function truncates to milliseconds.
    
    Returns:
        datetime with millisecond precision and UTC timezone
        
    Example:
        >>> ts = get_timestamp_ms()
        >>> # Use in LanceDB insert
        >>> data = {"created_at": ts, "updated_at": ts}
    """
    now = datetime.now(timezone.utc)
    # Truncate microseconds to milliseconds (remove last 3 digits)
    # Example: 123456 microseconds -> 123000 microseconds
    return now.replace(microsecond=now.microsecond // 1000 * 1000)


__all__ = [
    'get_timestamp_ms',
]
