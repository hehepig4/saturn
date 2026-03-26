"""
Unified truncation configuration for data values in LLM prompts.

IMPORTANT:
- TBox descriptions (class descriptions, CQs) should NEVER be truncated.
- Only data-related content (sample values, readouts, cell values) may be truncated.

Usage:
    from config.truncation_limits import TruncationLimits
    
    truncated = value[:TruncationLimits.SAMPLE_VALUE_MAX_LENGTH] + "..."
"""


class TruncationLimits:
    """
    Centralized limits for data-related content in LLM prompts.
    
    These limits prevent token overflow when processing tables with
    extremely long cell values (e.g., GeoJSON, base64, URLs).
    
    DO NOT use these limits for TBox/ontology content.
    """
    
    # ============ Sample Values ============
    # Maximum length per sample value shown in prompts
    SAMPLE_VALUE_MAX_LENGTH: int = 100  # Default was 50, doubled
    
    # Number of sample values to show per column
    SAMPLE_VALUE_COUNT: int = 3
    
    # ============ Readout / Data Property Summaries ============
    # Maximum length per facet readout (e.g., "Range: 1-100, Mean: 50")
    READOUT_MAX_LENGTH: int = 400  # Default was ~200, doubled
    
    # ============ Column Summary (combined) ============
    # Maximum total length for a single column's summary in annotation prompts
    # Includes: column name, class, all readouts, samples
    COLUMN_SUMMARY_MAX_LENGTH: int = 600
    
    # ============ Changelog / Iteration Logs ============
    # Maximum length for iteration changelog (soft limit)
    CHANGELOG_MAX_LENGTH: int = 2400  # Default was 1200, doubled
    
    # ============ Column Batch Processing ============
    # Maximum columns per single LLM call for classification/annotation
    # When a table has more columns than this, it will be processed in batches
    # Note: Reduced from 64 to 32 for better reliability with high-column tables
    MAX_COLUMNS_PER_LLM_CALL: int = 32

    # ============ LLM Call Configuration ============
    # Default timeout for LLM API calls (seconds)
    # Used across all LLM invocation functions for consistency
    LLM_TIMEOUT: float = 300.0  # 5 minutes


def truncate_value(value: str, max_length: int, ellipsis: str = "...") -> str:
    """
    Truncate a string value if it exceeds max_length.
    
    Args:
        value: The string to potentially truncate
        max_length: Maximum allowed length
        ellipsis: String to append if truncated (default "...")
        
    Returns:
        Original string if within limit, otherwise truncated with ellipsis
    """
    if len(value) <= max_length:
        return value
    return value[:max_length - len(ellipsis)] + ellipsis


def truncate_sample_values(
    values: list,
    n: int = TruncationLimits.SAMPLE_VALUE_COUNT,
    max_length: int = TruncationLimits.SAMPLE_VALUE_MAX_LENGTH,
) -> str:
    """
    Sample and truncate values for LLM prompt display.
    
    Args:
        values: List of values to sample from
        n: Number of samples to take
        max_length: Maximum length per value
        
    Returns:
        Formatted string like: "value1", "value2", "value3"
    """
    if not values:
        return ""
    
    # Take first n non-empty values
    samples = []
    for v in values:
        if v is not None:
            v_str = str(v).strip()
            if v_str:
                samples.append(truncate_value(v_str, max_length))
                if len(samples) >= n:
                    break
    
    return ", ".join(f'"{v}"' for v in samples)
