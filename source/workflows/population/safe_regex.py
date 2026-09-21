"""
Safe Regex Utilities (RE2-based)

Uses Google RE2 for all regex operations, guaranteeing O(n) linear time
matching and eliminating catastrophic backtracking (ReDoS) by design.

This module provides:
1. safe_compile() - Compile regex using RE2 (returns None on unsupported pattern)
2. safe_pandas_match() - Batch match using RE2 with error handling
3. validate_regex_safety() - Validate pattern compiles with RE2 (always safe if it compiles)

Usage:
    from workflows.population.safe_regex import safe_compile, safe_pandas_match

    matched_series, has_error = safe_pandas_match(pattern, series)
    if has_error:
        # Handle as failure - treat all values as unmatched
        pass
"""

from typing import List, Optional, Tuple
import pandas as pd
import re2
from loguru import logger


def validate_regex_safety(pattern: str, values: List[str] = None, **kwargs) -> bool:
    """
    Validate regex pattern is safe by compiling with RE2.

    RE2 guarantees linear-time matching, so any pattern that compiles
    is inherently safe from catastrophic backtracking.

    Patterns using features unsupported by RE2 (backreferences, lookahead/
    lookbehind) will fail compilation and be rejected.

    Args:
        pattern: Regex pattern to test
        values: Ignored (kept for API compatibility)

    Returns:
        True if pattern compiles with RE2, False otherwise
    """
    return safe_compile(pattern) is not None


def safe_compile(pattern: str) -> Optional[re2._Regexp]:
    """
    Safely compile a regex pattern using RE2.

    RE2 rejects patterns with backreferences, lookahead, lookbehind
    and other features that can cause exponential backtracking.

    Args:
        pattern: Regex pattern string

    Returns:
        Compiled RE2 pattern or None if pattern is invalid/unsupported
    """
    try:
        return re2.compile(pattern)
    except Exception as e:
        logger.debug(f"RE2 rejected pattern '{pattern[:80]}': {e}")
        return None


def safe_pandas_match(
    pattern: str,
    series: pd.Series,
) -> Tuple[pd.Series, bool]:
    """
    Perform batch regex match using RE2.

    Uses RE2 compiled pattern with Series.map() for safe linear-time matching.

    Args:
        pattern: Regex pattern string
        series: Pandas Series of strings to match

    Returns:
        Tuple of (matched_series, has_error)
        - matched_series: Boolean Series indicating matches (all False if error)
        - has_error: True if any error occurred
    """
    if series.empty:
        return (pd.Series([], dtype=bool), False)

    compiled = safe_compile(pattern)
    if compiled is None:
        logger.warning(f"RE2 rejected pattern in safe_pandas_match: {pattern[:80]}")
        return (pd.Series([False] * len(series), dtype=bool), True)

    try:
        str_series = series.astype("string")
        matched = str_series.map(
            lambda x: bool(compiled.match(str(x))) if pd.notna(x) else False
        ).astype(bool)
        return (matched, False)
    except Exception as e:
        logger.warning(f"RE2 match error for pattern '{pattern[:80]}': {e}")
        return (pd.Series([False] * len(series), dtype=bool), True)
