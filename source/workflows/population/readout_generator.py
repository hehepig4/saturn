"""
Readout Generator for Column DataProperty

Generates natural language descriptions from statistics using readout templates.
This bridges the gap between computed statistics and human-readable summaries.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from loguru import logger


@dataclass
class ReadoutResult:
    """Result of readout generation."""
    readout: str
    template_used: str
    placeholders_filled: Dict[str, Any]
    missing_placeholders: List[str]
    success: bool


class ReadoutGenerator:
    """
    Generates natural language readouts from column statistics.
    
    Takes a readout template with placeholders and fills them with
    actual statistics values.
    
    Example:
        template: "Year range: {min}-{max}"
        statistics: {"min": 1990, "max": 2024}
        result: "Year range: 1990-2024"
    """
    
    # Formatting rules for different statistic types
    FORMAT_RULES = {
        # Numeric statistics
        "min": lambda v: _format_value(v),
        "max": lambda v: _format_value(v),
        "mean": lambda v: f"{v:.2f}" if isinstance(v, float) else str(v),
        "median": lambda v: f"{v:.2f}" if isinstance(v, float) else str(v),
        "std": lambda v: f"{v:.2f}" if isinstance(v, float) else str(v),
        "sum": lambda v: _format_number(v),
        "count": lambda v: _format_number(v),
        "distinct_count": lambda v: _format_number(v),
        "null_ratio": lambda v: f"{v*100:.1f}%",
        # Percentiles and mode (new)
        "percentile_25": lambda v: _format_value(v),
        "percentile_75": lambda v: _format_value(v),
        "mode": lambda v: _format_value(v),
        # DateTime
        "range_days": lambda v: f"{v:,} days" if isinstance(v, (int, float)) else str(v),
        # String statistics
        "avg_length": lambda v: f"{v:.1f}" if isinstance(v, float) else str(v),
        "max_length": lambda v: _format_number(v),
        "word_count_avg": lambda v: f"{v:.1f} words" if isinstance(v, float) else str(v),
        "top_values": lambda v: _format_top_values(v),
        "sample_values": lambda v: _format_sample_values(v),
        "tfidf_keywords": lambda v: _format_keywords(v),
        # URI
        "domain_distribution": lambda v: _format_top_values(v),
    }
    
    def render(
        self,
        template: str,
        statistics: Dict[str, Any],
        datatype: Optional[str] = None,
        fallback_on_missing: bool = True,
    ) -> ReadoutResult:
        """
        Fill readout template with statistics values.
        
        Args:
            template: Template string with {placeholder} syntax
            statistics: Dict of computed statistics
            datatype: DataProperty range type (for validation)
            fallback_on_missing: If True, use "N/A" for missing values
            
        Returns:
            ReadoutResult with filled template and metadata
        """
        if not template:
            return ReadoutResult(
                readout="",
                template_used=template,
                placeholders_filled={},
                missing_placeholders=[],
                success=False,
            )
        
        # Extract placeholders from template
        placeholders = set(re.findall(r'\{(\w+)\}', template))
        
        # Track what we fill
        filled = {}
        missing = []
        
        # Build result string
        result = template
        
        for ph in placeholders:
            if ph in statistics and statistics[ph] is not None:
                # Get value and format it
                value = statistics[ph]
                formatter = self.FORMAT_RULES.get(ph, lambda v: str(v))
                
                try:
                    formatted = formatter(value)
                    result = result.replace(f"{{{ph}}}", formatted)
                    filled[ph] = value
                except Exception as e:
                    logger.warning(f"Failed to format {ph}={value}: {e}")
                    if fallback_on_missing:
                        result = result.replace(f"{{{ph}}}", "N/A")
                    missing.append(ph)
            else:
                missing.append(ph)
                if fallback_on_missing:
                    result = result.replace(f"{{{ph}}}", "N/A")
        
        success = len(missing) == 0
        
        return ReadoutResult(
            readout=result,
            template_used=template,
            placeholders_filled=filled,
            missing_placeholders=missing,
            success=success,
        )
    



# ============== Helper Functions ==============

def _format_value(v: Any) -> str:
    """Format a generic value for display."""
    if isinstance(v, float):
        if v.is_integer():
            return f"{int(v):,}"
        return f"{v:,.2f}"
    elif isinstance(v, int):
        return f"{v:,}"
    elif isinstance(v, str):
        return v
    else:
        return str(v)


def _format_number(v: Any) -> str:
    """Format a number with thousands separator."""
    if isinstance(v, (int, float)):
        if isinstance(v, float) and not v.is_integer():
            return f"{v:,.2f}"
        return f"{int(v):,}"
    return str(v)


def _format_top_values(v: Any) -> str:
    """Format top values list for display."""
    if not v:
        return "(none)"
    
    if isinstance(v, list):
        # Assume list of {"value": ..., "count": ...} dicts
        items = []
        for item in v[:5]:  # Limit to 5
            if isinstance(item, dict):
                val = item.get("value", "")
                items.append(str(val))
            else:
                items.append(str(item))
        return ", ".join(items)
    
    return str(v)


def _format_sample_values(v: Any) -> str:
    """Format sample values for display (deduplicated and truncated).
    
    Each value is truncated to prevent prompt overflow from binary data
    or extremely long strings.
    """
    # Import truncation limit
    from config.truncation_limits import TruncationLimits
    max_len = TruncationLimits.SAMPLE_VALUE_MAX_LENGTH
    
    if not v:
        return "(none)"
    
    if isinstance(v, list):
        # Deduplicate while preserving order
        seen = set()
        unique_samples = []
        for x in v:
            s = str(x)
            # Truncate each value to prevent binary/long data overflow
            if len(s) > max_len:
                s = s[:max_len] + "..."
            if s not in seen:
                seen.add(s)
                unique_samples.append(s)
        return ", ".join(unique_samples[:5])
    
    # Handle string case (e.g., already joined string)
    s = str(v)
    if len(s) > max_len * 5:  # Allow longer for pre-joined strings
        s = s[:max_len * 5] + "..."
    return s


def _format_keywords(v: Any) -> str:
    """Format TF-IDF keywords for display."""
    if not v:
        return "(none)"
    
    if isinstance(v, list):
        # List of keywords or keyword-score pairs
        items = []
        for item in v[:5]:  # Limit to 5
            if isinstance(item, dict):
                kw = item.get("keyword", item.get("term", ""))
                items.append(str(kw))
            elif isinstance(item, tuple):
                items.append(str(item[0]))  # (keyword, score) tuple
            else:
                items.append(str(item))
        return ", ".join(items)
    
    return str(v)


# ============== Convenience Functions ==============

def generate_readout(
    template: str,
    statistics: Dict[str, Any],
    datatype: Optional[str] = None,
) -> str:
    """
    Convenience function to generate a readout.
    
    Args:
        template: Readout template with {placeholders}
        statistics: Computed statistics dict
        datatype: Optional datatype for validation
        
    Returns:
        Filled readout string
    """
    generator = ReadoutGenerator()
    result = generator.render(template, statistics, datatype)
    return result.readout



